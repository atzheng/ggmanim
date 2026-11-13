"""Abstract :class:`Scale` base shared by continuous and discrete scales.

A scale owns one or more aesthetics (e.g. ``("x", "xmin", "xmax", ...)`` or
``("colour", "color")``). Its life cycle inside a build is::

    s = spec_scale.clone()        # untrained copy; the plot spec is never mutated
    s.train(values)               # once per layer/column that uses the aesthetic
    s.map(values)                 # data -> scale output (positions, colours, ...)
    s.get_limits(); s.get_breaks(); s.get_labels()   # for axes / legends

Subclasses implement :meth:`train`, :meth:`map`, :meth:`get_limits`,
:meth:`get_breaks`, :meth:`get_labels` and :meth:`reset`. ``reset()`` must clear
*only* trained state (never user configuration), so that :meth:`clone` -- a shallow
copy followed by ``reset()`` -- yields an independent untrained scale. Subclasses
whose trained state is a mutable container must assign a fresh container in
``reset()`` rather than mutating it in place.

No manim imports here.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Iterable

__all__ = ["Scale"]


class Scale(ABC):
    """Base class for all scales.

    Attributes
    ----------
    aesthetics : tuple[str, ...]
        Aesthetics this scale applies to. The first one is the "primary" aesthetic
        (used for e.g. default titles).
    name : str | None
        Title for the axis / legend. ``None`` means "not set": the plot falls back
        to ``labs()`` or the mapped column name.
    is_discrete : bool
        Class attribute; ``False`` for continuous scales, ``True`` for discrete ones.
    """

    is_discrete: bool = False

    def __init__(self, aesthetics: str | Iterable[str], name: str | None = None) -> None:
        if isinstance(aesthetics, str):
            aesthetics = (aesthetics,)
        self.aesthetics: tuple[str, ...] = tuple(aesthetics)
        if not self.aesthetics:
            raise ValueError("a scale needs at least one aesthetic")
        self.name = name

    # ------------------------------------------------------------------ abstract
    @abstractmethod
    def train(self, values: Any) -> None:
        """Update the trained range with ``values`` (array-like, in data space).
        May be called many times; ranges accumulate. Missing values are ignored."""

    @abstractmethod
    def map(self, values: Any) -> Any:
        """Map data-space ``values`` to scale output (a numpy array or list of
        the same length). Values outside the limits follow the scale's ``oob``
        policy (missing / NaN by default)."""

    @abstractmethod
    def get_limits(self) -> Any:
        """Limits of the scale: user limits if given, else the trained range.
        Continuous: ``(lo, hi)`` in transformed space; discrete: list of levels.
        ``None`` if the scale has neither been trained nor given limits."""

    @abstractmethod
    def get_breaks(self) -> Any:
        """Break positions (continuous: transformed space; discrete: levels)."""

    @abstractmethod
    def get_labels(self) -> list[str]:
        """One label string per break returned by :meth:`get_breaks`."""

    @abstractmethod
    def reset(self) -> None:
        """Forget all trained state (keep user configuration)."""

    # ---------------------------------------------------------------- concrete
    @property
    def aesthetic(self) -> str:
        """The primary aesthetic (``aesthetics[0]``)."""
        return self.aesthetics[0]

    def has_aesthetic(self, aes: str) -> bool:
        return aes in self.aesthetics

    def is_empty(self) -> bool:
        """True if the scale has no limits (not trained and none given)."""
        return self.get_limits() is None

    def clone(self) -> "Scale":
        """Untrained copy with the same configuration (shallow copy + reset)."""
        new = copy.copy(self)
        new.reset()
        return new

    def add_to(self, plot):
        """Component protocol: ``plot.add(scale)`` calls this."""
        return plot._with_scale(self)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} aesthetics={self.aesthetics!r} name={self.name!r}>"
