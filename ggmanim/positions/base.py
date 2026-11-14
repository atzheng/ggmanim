"""Position base class, :class:`PositionIdentity` and the position registry.

A *position adjustment* moves the rows of a layer after the stat and after
``geom.setup_data`` (ggplot2 order), e.g. stacking or dodging bars. Positions are
pure pandas/numpy and never import manim.

Pipeline contract (``build.apply_positions``)
---------------------------------------------
For every layer, with ``pos = layer.position`` (always a :class:`Position`, see
:func:`get_position`)::

    params = pos.setup_params(data)
    data = pos.setup_data(data, params)
    data = pos.compute_layer(data, params, scales)

``data`` is the layer data after ``geom.setup_data``: x-like/y-like columns are in
*transformed* space (``x``, ``xmin``, ``xmax``, ``y``, ``ymin``, ``ymax`` ...),
non-position aesthetics are already mapped, and ``group`` and ``PANEL`` columns are
present (for non-empty layers). ``scales`` is ``{"x": scale, "y": scale}`` (the
build's position scales, trained pre-stat; read them, never train them). After
every layer's position, the pipeline resets and retrains the position scales on
the adjusted data, so a position may move rows anywhere.

:meth:`Position.compute_layer` splits by ``PANEL`` and calls
:meth:`Position.compute_panel` once per panel; most positions override only
``compute_panel`` (and ``setup_params`` to resolve e.g. a default width). Return a
new DataFrame; never modify ``data`` in place.

Adding a position
-----------------
Create ``ggmanim/positions/<name>.py`` with a :class:`Position` subclass that sets
the class attribute ``name`` (``"stack"``) and a ``position_<name>(**params)``
constructor, both listed in the module's ``__all__``; then run
``python scripts/gen_exports.py``. Subclasses with a ``name`` register themselves in
:data:`POSITIONS` on class creation, so ``geom_col(position="stack")`` resolves
through :func:`get_position` without editing any other file.
"""

from __future__ import annotations

import ast
from typing import Any, ClassVar

import pandas as pd

__all__ = [
    "POSITIONS",
    "Position",
    "PositionIdentity",
    "get_position",
    "parse_position",
    "position_identity",
]

#: name -> Position subclass. Filled by :meth:`Position.__init_subclass__`.
POSITIONS: dict[str, type[Position]] = {}


class Position:
    """Base position adjustment.

    Attributes
    ----------
    name : str
        Registry key (``"identity"``, ``"stack"`` ...). A subclass that defines
        ``name`` in its class body is registered in :data:`POSITIONS`.
    required_aes : tuple
        Columns that must be present in the data (checked by :meth:`compute_layer`).
    params : dict
        Constructor arguments (``position_dodge(width=0.5)`` -> ``{"width": 0.5}``).
    """

    name: ClassVar[str] = "position"
    required_aes: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "name" in cls.__dict__:
            POSITIONS[cls.name] = cls

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Position)
            and type(other) is type(self)
            and other.params == self.params
        )

    __hash__ = None  # type: ignore[assignment]

    # -- pipeline hooks -------------------------------------------------------
    def setup_params(self, data: pd.DataFrame) -> dict[str, Any]:
        """Parameters for this layer (may look at ``data``). Default: a copy of
        :attr:`params`."""
        return dict(self.params)

    def setup_data(self, data: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Check :attr:`required_aes` and prepare the data. Default: unchanged."""
        missing = [a for a in self.required_aes if a not in data.columns]
        if missing and len(data):
            raise ValueError(
                f"{type(self).__name__} requires the following missing aesthetics: "
                + ", ".join(missing)
            )
        return data

    def compute_layer(
        self, data: pd.DataFrame, params: dict[str, Any], scales: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Split by ``PANEL`` (if present) and call :meth:`compute_panel` per panel."""
        if len(data) == 0:
            return data
        if "PANEL" not in data.columns:
            return self.compute_panel(data, params, scales)
        pieces = [
            self.compute_panel(pdf, params, scales)
            for _, pdf in data.groupby("PANEL", sort=True, observed=True)
        ]
        if len(pieces) == 1:
            return pieces[0]
        return pd.concat(pieces).sort_index(kind="stable")

    def compute_panel(
        self, data: pd.DataFrame, params: dict[str, Any], scales: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Adjust one panel's rows. Subclasses override this."""
        raise NotImplementedError


class PositionIdentity(Position):
    """No adjustment: the data is returned as is (the same object)."""

    name = "identity"

    def compute_layer(
        self, data: pd.DataFrame, params: dict[str, Any], scales: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        return data

    def compute_panel(
        self, data: pd.DataFrame, params: dict[str, Any], scales: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        return data


def position_identity() -> PositionIdentity:
    """Don't adjust positions (ggplot2's ``position_identity()``)."""
    return PositionIdentity()


def parse_position(spec: str) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
    """Split a position string into ``(name, args, kwargs)``.

    ``"dodge"`` -> ``("dodge", (), {})``; ``"dodge(width=0.8)"`` ->
    ``("dodge", (), {"width": 0.8})``; ``"position_nudge(0, y=-0.1)"`` ->
    ``("position_nudge", (0,), {"y": -0.1})``. Arguments must be Python literals
    (numbers, strings, booleans, ``None``, tuples/lists/dicts of those): they are
    read with :func:`ast.literal_eval`, never evaluated. Raises ``ValueError``
    for anything else.
    """
    try:
        node = ast.parse(spec.strip(), mode="eval").body
    except SyntaxError:
        raise ValueError(f"Cannot parse position {spec!r}") from None
    if isinstance(node, ast.Name):
        return node.id, (), {}
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
        raise ValueError(f"Cannot parse position {spec!r}: expected 'name' or 'name(arg=...)'")
    try:
        args = tuple(ast.literal_eval(a) for a in node.args)
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise ValueError("**kwargs")
            kwargs[kw.arg] = ast.literal_eval(kw.value)
    except ValueError:
        raise ValueError(
            f"Cannot parse position {spec!r}: arguments must be literals (numbers, strings, "
            "True/False/None, ...)"
        ) from None
    return node.func.id, args, kwargs


def get_position(position: Any = None, **params: Any) -> Position:
    """Resolve ``position`` to a :class:`Position` instance.

    Accepts ``None`` (identity), a :class:`Position` instance (returned as is;
    ``params`` must then be empty), a :class:`Position` subclass (instantiated with
    ``params``), or a name such as ``"stack"`` / ``"position_stack"`` looked up in
    :data:`POSITIONS`. A name may carry literal arguments, ``"dodge(width=0.8)"``
    or ``"position_jitter(0.1, seed=1)"`` (parsed safely, see
    :func:`parse_position`); a keyword given both there and in ``params`` is a
    ``TypeError``. Raises ``ValueError`` for an unknown name or an unparsable
    string and ``TypeError`` for anything else.
    """
    if position is None:
        position = "identity"
    if isinstance(position, Position):
        if params:
            raise TypeError(f"cannot pass parameters to an existing {position!r}")
        return position
    if isinstance(position, type) and issubclass(position, Position):
        return position(**params)
    if isinstance(position, str):
        name, args, kwargs = parse_position(position)
        key = name[9:] if name.startswith("position_") else name
        cls = POSITIONS.get(key)
        if cls is None:
            raise ValueError(
                f"Unknown position {position!r}; available: {', '.join(sorted(POSITIONS))}"
            )
        both = sorted(set(kwargs) & set(params))
        if both:
            raise TypeError(f"position {position!r}: {', '.join(both)} given twice")
        return cls(*args, **kwargs, **params)
    raise TypeError(f"Cannot interpret {position!r} as a position")
