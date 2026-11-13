"""Geom base class and layer construction helpers.

A *geom* turns a layer's computed data into manim mobjects. Geoms are stateless:
everything layer-specific (constant aesthetics, geom/stat parameters) lives in
:attr:`ggmanim.layer.Layer.params` and is passed to the methods below.

Contract with the build pipeline (``ggm-s11.8``)
-----------------------------------------------
For each layer ``L`` (``geom = L.geom``, ``stat = L.stat``, ``params = L.params``):

1. Resolve data/mapping (``L.resolve_data`` / ``L.resolve_mapping``), evaluate the
   aesthetics, ``add_group``.
2. *Pre-stat* position training: for every ``aes, cols`` in
   ``geom.position_aes_for_training(df).items()`` train the ``aes`` (``"x"``/``"y"``)
   scale on ``df[cols]``. Only columns that exist are returned, so a
   ``geom_function`` layer (no data) contributes nothing, ``geom_vline`` only its
   ``xintercept`` and ``geom_hline`` only its ``yintercept``.
3. Map x-like / y-like columns to transformed space (``scale.transform``), then
   ``df = stat.compute_layer(df, scales, **stat.params)``.
4. Map non-position aesthetics (``color``/``fill`` -> ``ManimColor`` via the discrete
   scales).
5. ``df = geom.setup_data(df, params)`` (e.g. ``geom_col`` adds
   ``xmin``/``xmax``/``ymin``/``ymax`` in transformed space).
6. Reset + re-train position scales on ``geom.position_aes_for_training(df)`` for
   every layer (now ``geom_function``/``geom_smooth`` contribute their curves and
   ribbons, ``geom_col`` its bar extents including ``ymin = 0``), then apply oob.
7. ``panel = coord.build(x_scale, y_scale, theme, labels)``.
8. ``mob = geom.draw_panel(df, panel.panel_params, panel, theme, **params)``.

``draw_panel`` receives data whose x-like/y-like columns are in *transformed*
space, ``color``/``fill`` either already ``ManimColor`` (mapped) or absent, and a
``group`` column (added if missing). Constant aesthetics in ``params``
(``geom_point(color=RED)``) override mapped and default values. Default aesthetic
values that are ``None`` are resolved per geom (``color=None`` usually means the
theme's foreground colour; for ``geom_col`` it means "no outline").

Legend guides (``ggm-s11.7``) call ``geom.draw_key(row, theme)`` with a dict of
aesthetic values (e.g. ``{"color": ManimColor(...)}``); missing aesthetics fall back
to the geom defaults. The key is centred on the origin and about
:data:`KEY_SIZE` scene units across.

Aesthetic -> manim translation (Phase 1)
----------------------------------------
``color`` -> stroke colour (``Dot``: fill colour), ``fill`` -> ``fill_color``,
``alpha`` -> ``fill_opacity``/``stroke_opacity``, ``size`` -> point radius (scene
units), ``linewidth`` -> ``stroke_width`` (manim units), ``linetype`` -> solid only.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from manim import ManimColor, Mobject, VGroup, VMobject

from ..aes import AES_ALIASES, Aes, standardise_aes_name
from ..layer import Layer
from ..palettes import as_manim_color
from ..scales.continuous import X_AESTHETICS, Y_AESTHETICS
from ..stats import Stat, add_group, get_stat
from ..stats import STATS
from ..theme import Theme, theme_default

__all__ = [
    "Geom",
    "KEY_SIZE",
    "make_layer",
    "resolve_color",
    "group_label",
]

#: Approximate width/height of a legend key glyph in scene units.
KEY_SIZE = 0.3

#: Aesthetic names a geom may receive as constant params (besides its defaults).
_KNOWN_AES = frozenset(
    {"x", "y", "color", "fill", "alpha", "size", "linewidth", "linetype", "shape", "group"}
    | set(X_AESTHETICS)
    | set(Y_AESTHETICS)
)

#: Params every layer accepts.
_COMMON_PARAMS = frozenset({"na_rm", "show_legend"})


def resolve_color(value: Any, fallback: Any = None) -> Optional[ManimColor]:
    """``value`` as a ``ManimColor``; ``None``/NaN -> ``fallback`` (also coerced)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        if fallback is None:
            return None
        value = fallback
    return as_manim_color(value)


def _scalar(v: Any) -> Any:
    """NaN -> None, numpy scalars -> python."""
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


class Geom:
    """Base geom. Subclasses set the class attributes and implement
    :meth:`draw_panel` (or :meth:`draw_group`) and :meth:`draw_key`.

    Attributes
    ----------
    required_aes : tuple
        Columns that must be present in the data given to ``draw_panel``.
    default_aes : dict
        Default aesthetic values (manim units; see module docstring). Every key
        is also a valid constant param (``geom_point(size=0.1)``).
    extra_params : tuple
        Non-aesthetic geom params (``label``, ``width``, ``se`` ...).
    default_stat : str
        Stat used when the constructor is given ``stat=None``.
    x_aes, y_aes : tuple
        Columns that, when present, train the x / y position scale (see
        :meth:`position_aes_for_training`).
    """

    required_aes: Tuple[str, ...] = ()
    default_aes: Dict[str, Any] = {}
    extra_params: Tuple[str, ...] = ()
    default_stat: str = "identity"
    x_aes: Tuple[str, ...] = X_AESTHETICS
    y_aes: Tuple[str, ...] = Y_AESTHETICS

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    # -- parameters -----------------------------------------------------------
    def aesthetics(self) -> frozenset:
        """All aesthetic names this geom understands."""
        return frozenset(self.required_aes) | frozenset(self.default_aes) | _KNOWN_AES

    def aes_params(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """The constant aesthetics in ``params`` (names standardised)."""
        out = {}
        aes = self.aesthetics()
        for k, v in params.items():
            k = standardise_aes_name(k)
            if k in aes and v is not None:
                out[k] = v
        return out

    # -- pipeline hooks -------------------------------------------------------
    def position_aes_for_training(self, data: Optional[pd.DataFrame]) -> Dict[str, list]:
        """``{"x": [cols], "y": [cols]}``: the columns of ``data`` (those present)
        that train the x and y position scales. Call before the stat (raw layer
        data) and again after :meth:`setup_data`."""
        cols = set(data.columns) if data is not None else set()
        return {
            "x": [c for c in self.x_aes if c in cols],
            "y": [c for c in self.y_aes if c in cols],
        }

    def setup_data(self, data: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        """Hook run after the stat (data in transformed space), before the final
        position-scale training. Default: returns ``data`` unchanged."""
        return data

    def check_required_aes(self, data: pd.DataFrame) -> None:
        missing = [a for a in self.required_aes if a not in data.columns]
        if missing:
            raise ValueError(
                f"{type(self).__name__} requires the following missing aesthetics: "
                + ", ".join(missing)
            )

    def use_defaults(self, data: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        """``data`` with every :attr:`default_aes` column filled in (where absent)
        and constant aesthetics from ``params`` overriding mapped values."""
        out = data.copy()
        n = len(out)
        for k, v in self.default_aes.items():
            if k not in out.columns:
                out[k] = pd.Series([v] * n, index=out.index, dtype=object)
        for k, v in self.aes_params(params).items():
            if k in ("x", "y", "group") or k in self.x_aes or k in self.y_aes:
                continue  # position constants are handled by the constructor (data)
            out[k] = pd.Series([v] * n, index=out.index, dtype=object)
        if "group" not in out.columns:
            out["group"] = _group_ids(data)
        return out

    # -- drawing --------------------------------------------------------------
    def draw_panel(self, data: pd.DataFrame, panel_params, coord, theme: Optional[Theme] = None,
                   **params) -> VMobject:
        """Draw the whole layer for one panel; default: :meth:`draw_group` per
        group, in ``group`` order, collected in a ``VGroup``.

        ``coord`` is the built :class:`~ggmanim.coords.CartesianPanel` (anything with
        ``transform(x, y) -> (N, 3)`` taking transformed-space values).
        """
        theme = theme if theme is not None else theme_default()
        self.check_required_aes(data)
        data = self.use_defaults(data, params)
        out = VGroup()
        for _, gdf in data.groupby("group", sort=True):
            out.add(self.draw_group(gdf, panel_params, coord, theme, **params))
        return out

    def draw_group(self, data: pd.DataFrame, panel_params, coord, theme: Theme,
                   **params) -> VMobject:
        raise NotImplementedError

    def draw_key(self, data_row: Mapping[str, Any], theme: Optional[Theme] = None) -> VMobject:
        """A small legend glyph (about :data:`KEY_SIZE` across, centred at origin)."""
        raise NotImplementedError

    def key_aes(self, data_row: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """``data_row`` (standardised names) over :attr:`default_aes`."""
        out = dict(self.default_aes)
        for k, v in (data_row or {}).items():
            v = _scalar(v)
            if v is not None:
                out[standardise_aes_name(k)] = v
        return out


# -- helpers for subclasses -------------------------------------------------------


def _group_ids(data: pd.DataFrame) -> np.ndarray:
    """``add_group`` ids for data that may already hold mapped ``ManimColor`` values
    (which are not orderable): colours are compared by their hex string."""
    keyed = data.copy()
    for c in keyed.columns:
        if keyed[c].dtype == object:
            keyed[c] = keyed[c].map(lambda v: v.to_hex() if isinstance(v, ManimColor) else v)
    return add_group(keyed)["group"].to_numpy()


def transform_points(coord, x, y) -> np.ndarray:
    """``coord.transform(x, y)`` (also accepts a bare manim ``Axes``)."""
    if hasattr(coord, "transform"):
        return np.asarray(coord.transform(x, y), dtype=float).reshape(-1, 3)
    xs, ys = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    pts = np.asarray(coord.c2p(xs.ravel(), ys.ravel()), dtype=float)
    return pts.reshape(3, -1).T


def num(v: Any, default: float) -> float:
    """``float(v)`` or ``default`` for ``None``/NaN/non-numeric."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def path_mobject(points: np.ndarray, color: Any, linewidth: float, alpha: float) -> VMobject:
    """An open polyline through ``points`` built with ``set_points_as_corners``."""
    m = VMobject()
    if len(points):
        m.set_points_as_corners(points)
    m.set_fill(opacity=0)
    m.set_stroke(color=color, width=linewidth, opacity=alpha)
    return m


def assemble(marks: VGroup, labels: VGroup) -> VGroup:
    """``VGroup(*marks, *labels)`` with ``.marks`` and ``.labels`` attributes."""
    out = VGroup(*marks, *labels)
    out.marks = marks
    out.labels = labels
    return out


def first(data: pd.DataFrame, col: str, default: Any = None) -> Any:
    """First value of ``data[col]`` (NaN -> ``default``)."""
    if col not in data.columns or len(data) == 0:
        return default
    v = _scalar(data[col].iloc[0])
    return default if v is None else v


def finite_rows(data: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Rows of ``data`` whose ``cols`` are all finite numbers (oob-censored rows
    are dropped, like ggplot2's ``remove_missing``)."""
    mask = np.ones(len(data), dtype=bool)
    for c in cols:
        if c in data.columns:
            mask &= np.isfinite(pd.to_numeric(data[c], errors="coerce").to_numpy(dtype=float))
    return data[mask]


def group_label(label: Any, index: int, group_id: Any) -> Any:
    """The label for the ``index``-th group (id ``group_id``).

    ``label`` may be ``None``, a string / Mobject (every group), a dict keyed by
    group id, or a sequence indexed by group order.
    """
    if label is None or isinstance(label, (str, Mobject)):
        return label
    if isinstance(label, Mapping):
        return label.get(group_id)
    if isinstance(label, Sequence):
        return label[index] if index < len(label) else None
    return str(label)


def make_label(theme: Theme, label: Any, color: Any) -> Optional[Mobject]:
    """Text (via the theme) or a copy of a ready-made Mobject, coloured ``color``."""
    if label is None:
        return None
    if isinstance(label, Mobject):
        mob = label.copy()
        if color is not None:
            mob.set_color(color)
        return mob
    if str(label) == "":
        return None
    return theme.make_text(str(label), font_size=theme.axis_text_size, color=color)


# -- layer construction -----------------------------------------------------------


def _stat_class(stat):
    if isinstance(stat, Stat):
        return type(stat)
    if isinstance(stat, type) and issubclass(stat, Stat):
        return stat
    if isinstance(stat, str):
        name = stat[5:] if stat.startswith("stat_") else stat
        return STATS.get(name)
    return None


def _stat_params(cls, params: Mapping[str, Any]) -> Dict[str, Any]:
    if cls is None:
        return {}
    names = set(cls.parameters())
    try:
        sig = inspect.signature(cls.__init__)
        names |= {n for n, p in sig.parameters.items()
                  if n != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    except (TypeError, ValueError):  # pragma: no cover
        pass
    return {k: v for k, v in params.items() if k in names}


def make_layer(geom: Geom, mapping=None, data=None, *, stat=None, position="identity",
               inherit_aes: bool = True, name: Optional[str] = None,
               params: Optional[Mapping[str, Any]] = None) -> Layer:
    """Build the :class:`~ggmanim.layer.Layer` for ``geom``.

    * ``mapping``/``data`` may be given in either order (a DataFrame passed as
      ``mapping`` is treated as data).
    * Param names are standardised (``colour``/``stroke_width``/``fill_color`` ...).
    * ``stat`` (name, class or instance; ``None`` -> ``geom.default_stat``) is
      instantiated with the params it accepts; ``Layer.params`` keeps *all* params
      (constant aesthetics, geom params and stat params).
    * Unknown params trigger a warning (ggplot2's "Ignoring unknown parameters").
    """
    if isinstance(mapping, pd.DataFrame) and (data is None or isinstance(data, Mapping)):
        mapping, data = data, mapping
    if mapping is not None and not isinstance(mapping, Aes):
        mapping = Aes(mapping)

    params = {standardise_aes_name(k) if k in AES_ALIASES else k: v
              for k, v in dict(params or {}).items()}

    stat = geom.default_stat if stat is None else stat
    cls = _stat_class(stat)
    sp = _stat_params(cls, params)
    stat_obj = get_stat(stat, **sp)

    known = geom.aesthetics() | set(geom.extra_params) | set(sp) | _COMMON_PARAMS
    unknown = sorted(k for k in params if k not in known)
    if unknown:
        warnings.warn(
            f"{type(geom).__name__}: ignoring unknown parameters: {', '.join(unknown)}",
            stacklevel=3,
        )

    return Layer(geom=geom, stat=stat_obj, position=position, data=data,
                 mapping=mapping, params=params, inherit_aes=inherit_aes, name=name)
