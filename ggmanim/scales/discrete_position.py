"""Discrete (categorical) x/y position scales: ``scale_x_discrete`` / ``scale_y_discrete``.

A discrete position scale maps the levels of a categorical variable to the
integer positions ``1..n`` on a continuous axis, so after
:meth:`ScaleDiscretePosition.transform` every position column is a float column
and all geoms, coords and stats work unchanged.

It exposes the same position-scale API as
:class:`~ggmanim.scales.continuous.ScaleContinuousPosition` (see the table at the
top of :mod:`ggmanim.scales.continuous`), with these discrete semantics:

``train(values)``            discrete values -> add levels; numeric values (e.g.
                             ``x = 1.5`` between two levels) -> continuous range.
``train_transformed(tv)``    numeric positions -> continuous range.
``transform(values)``        levels -> ``1..n`` (values outside the limits -> NaN,
                             with a warning); numeric values pass through.
``map`` / ``map_transformed``  same as ``transform`` / identity (no ``oob``).
``get_limits()``             the ordered list of levels (ggplot2 semantics).
``get_expanded_limits()``    ``(lo, hi)`` numeric axis range: ``(1, n)`` expanded by
                             ``expand`` (default ``add=0.6``) unioned with the
                             unexpanded continuous range (bar edges, ...).
``dimension()``              alias of ``get_expanded_limits()``.
``get_breaks(limits=None)``  float positions of the break levels, censored to the
                             numeric ``limits`` (default: the expanded range).
``get_labels(limits=None)``  one label per ``get_breaks(limits)``.

Like ggplot2's ``ScaleDiscretePosition``, :meth:`reset` forgets only the
continuous range (the build retrains positions after the stat and geom
``setup_data``, when the columns are already numeric); the levels survive.
:meth:`clone` clears both.

No manim imports here (the level helpers come from :mod:`.discrete`).
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..registry import register
from .base import Scale
from .breaks import expand_range, expansion
from .continuous import X_AESTHETICS, Y_AESTHETICS
from .discrete import _is_na, _sorted_levels, _to_python, is_discrete

__all__ = [
    "DEFAULT_DISCRETE_EXPANSION",
    "ScaleDiscretePosition",
    "scale_x_discrete",
    "scale_y_discrete",
]

DEFAULT_DISCRETE_EXPANSION = expansion(add=0.6)
"""ggplot2's default for discrete position scales: 0.6 level units each side."""

_UNSET = object()


def _is_categorical(values: Any) -> bool:
    return isinstance(values, pd.Categorical) or (
        isinstance(values, (pd.Series, pd.Index)) and isinstance(values.dtype, pd.CategoricalDtype)
    )


def _values_discrete(values: Any) -> bool:
    """True if ``values`` hold levels rather than numeric positions (bool counts as numeric)."""
    if _is_categorical(values):
        return True
    s = values if isinstance(values, pd.Series) else pd.Series(np.atleast_1d(values))
    if pd.api.types.is_bool_dtype(s.dtype):
        return False
    return len(s.dropna()) > 0 and is_discrete(s)


class ScaleDiscretePosition(Scale):
    """Discrete x or y position scale: levels -> integer positions ``1..n``.

    Parameters
    ----------
    aesthetics : str or sequence of str (e.g. :data:`.continuous.X_AESTHETICS`).
    name : axis title (``None`` = default from labs / mapping).
    limits : list of levels: fixes, reorders and subsets the levels. Data values
        not in ``limits`` are removed (mapped to NaN) with a warning.
    breaks : subset/order of levels that get a tick (default: all levels;
        ``[]`` for none) or a callable ``f(levels) -> levels``.
    labels : list (parallel to ``breaks`` if given, else to the levels), dict
        level -> label, or callable ``f(level) -> str``.
    expand : ``None`` (``add=0.6`` each side), ``False``, ``(mult, add)`` or
        ``(mult_lo, add_lo, mult_hi, add_hi)``.
    drop : drop unused categories of a pandas Categorical when training.

    Level order: a pandas Categorical's categories in category order (several
    Categoricals accumulate in encounter order); otherwise the levels are sorted,
    as ggplot2 does for character vectors.
    """

    is_discrete = True

    def __init__(
        self,
        aesthetics,
        name: str | None = None,
        limits: Sequence[Any] | None = None,
        breaks: Sequence[Any] | Callable | None = None,
        labels: Sequence[str] | Mapping[Any, str] | Callable | None = None,
        expand=None,
        drop: bool = True,
    ) -> None:
        super().__init__(aesthetics, name)
        self.limits = None if limits is None else [_to_python(v) for v in limits]
        self.breaks = breaks if breaks is None or callable(breaks) else list(breaks)
        if labels is not None and not callable(labels) and not isinstance(labels, Mapping):
            labels = [str(lab) for lab in labels]
            if isinstance(self.breaks, list) and len(labels) != len(self.breaks):
                raise ValueError("labels must have the same length as breaks")
        self.labels = labels
        self.expand = expand
        self.drop = drop
        self._levels: list[Any] = []
        self._ordered = False
        self._range_c: tuple[float, float] | None = None

    # ------------------------------------------------------------- training
    def reset(self) -> None:
        """Forget the continuous range only (levels are kept, as in ggplot2)."""
        self._range_c = None

    def clone(self) -> ScaleDiscretePosition:
        """Untrained copy (no levels, no continuous range) with the same configuration."""
        new = copy.copy(self)
        new._levels = []
        new._ordered = False
        new._range_c = None
        return new

    def train(self, values: Any) -> None:
        """Levels from discrete ``values``; numeric ``values`` train the continuous range."""
        if _is_categorical(values):
            cat = values.array if isinstance(values, pd.Series) else values
            if isinstance(cat, pd.Index):
                cat = cat.values
            cats = list(cat.categories)
            if self.drop:
                present = set(pd.Series(cat).dropna().unique())
                cats = [c for c in cats if c in present]
            self._extend(_to_python(c) for c in cats)
            self._ordered = True
            return
        if isinstance(values, (str, bytes)) or np.ndim(values) == 0:
            values = [values]
        if not _values_discrete(values):
            self.train_transformed(values)
            return
        s = values if isinstance(values, pd.Series) else pd.Series(list(values), dtype=object)
        self._extend(_to_python(v) for v in pd.unique(s.astype(object)) if not _is_na(v))
        if not self._ordered:
            self._levels = _sorted_levels(self._levels)

    def _extend(self, new: Iterable[Any]) -> None:
        seen = set(self._levels)
        for v in new:
            if v not in seen:
                self._levels.append(v)
                seen.add(v)

    def train_transformed(self, values: Any) -> None:
        """Update the continuous range with numeric positions (NaN/inf ignored)."""
        x = pd.to_numeric(pd.Series(np.atleast_1d(values)), errors="coerce").to_numpy(float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return
        lo, hi = float(x.min()), float(x.max())
        if self._range_c is not None:
            lo, hi = min(lo, self._range_c[0]), max(hi, self._range_c[1])
        self._range_c = (lo, hi)

    @property
    def range(self) -> tuple[float, float] | None:
        """Continuous range trained so far (``None`` if none)."""
        return self._range_c

    # ------------------------------------------------------------- mapping
    def transform(self, values: Any) -> np.ndarray:
        """Levels -> float positions ``1..n``; numeric values pass through.

        Discrete values not among the limits become NaN (a warning says how many).
        """
        if isinstance(values, (str, bytes)) or np.ndim(values) == 0:
            values = [values]
        if not _values_discrete(values):
            return pd.to_numeric(pd.Series(np.asarray(values)), errors="coerce").to_numpy(float)
        pos = {lvl: float(i) for i, lvl in enumerate(self.get_limits(), start=1)}
        vals = list(values)
        out = np.full(len(vals), np.nan)
        dropped = 0
        for i, v in enumerate(vals):
            if _is_na(v):
                continue
            v = _to_python(v)
            try:
                p = pos.get(v)
            except TypeError:  # unhashable
                p = None
            if p is None:
                dropped += 1
            else:
                out[i] = p
        if dropped:
            warnings.warn(
                f"Removed {dropped} value(s) outside the limits of the discrete "
                f"{self.aesthetic} scale",
                stacklevel=2,
            )
        return out

    def map(self, values: Any) -> np.ndarray:
        """Same as :meth:`transform` (discrete position scales have no ``oob``)."""
        return self.transform(values)

    def map_transformed(self, values: Any) -> np.ndarray:
        """Numeric positions, unchanged (as floats)."""
        return pd.to_numeric(pd.Series(np.atleast_1d(values)), errors="coerce").to_numpy(float)

    # ------------------------------------------------------------- limits
    def get_limits(self) -> list[Any]:
        """The ordered levels (user ``limits`` override training)."""
        return list(self.limits) if self.limits is not None else list(self._levels)

    def is_empty(self) -> bool:
        return not self.get_limits() and self._range_c is None

    def get_expanded_limits(self, expand=_UNSET) -> tuple[float, float] | None:
        """Numeric axis range: ``(1, n)`` expanded, unioned with the unexpanded
        continuous range (ggplot2's ``expand_limits_discrete``)."""
        if expand is _UNSET or expand is None or expand is True:
            expand = DEFAULT_DISCRETE_EXPANSION if self.expand is None else self.expand
        n = len(self.get_limits())
        if n == 0:
            if self._range_c is None:
                return None
            return expand_range(self._range_c, expand)
        lo, hi = expand_range((1.0, float(n)), expand)
        if self._range_c is not None:
            lo, hi = min(lo, self._range_c[0]), max(hi, self._range_c[1])
        return (lo, hi)

    def dimension(self, expand=_UNSET) -> tuple[float, float] | None:
        """ggplot2-style alias for :meth:`get_expanded_limits`."""
        return self.get_expanded_limits(expand)

    def continuous_limits(self) -> tuple[float, float] | None:
        """Numeric limits without expansion: ``(1, n)`` unioned with the continuous range."""
        return self.get_expanded_limits(False)

    # ------------------------------------------------------------- breaks/labels
    def get_break_levels(self) -> list[Any]:
        """The levels that get a tick, in axis order."""
        levels = self.get_limits()
        if self.breaks is None:
            return levels
        wanted = list(self.breaks(levels)) if callable(self.breaks) else self.breaks
        lim = set(levels)
        return [b for b in (_to_python(w) for w in wanted) if b in lim]

    def _breaks_and_labels(self, limits=None) -> tuple[np.ndarray, list[str]]:
        levels = self.get_limits()
        pos = {lvl: float(i) for i, lvl in enumerate(levels, start=1)}
        brk = self.get_break_levels()
        labels = self._labels_for(brk, levels)
        b = np.array([pos[x] for x in brk], dtype=float)
        if limits is not None and len(b):
            lo, hi = float(min(limits)), float(max(limits))
            keep = (b >= lo - 1e-9) & (b <= hi + 1e-9)
            b, labels = b[keep], [lab for lab, k in zip(labels, keep) if k]
        return b, labels

    def _labels_for(self, brk: list[Any], levels: list[Any]) -> list[str]:
        labels = self.labels
        if labels is None:
            return [str(x) for x in brk]
        if callable(labels):
            return [str(labels(x)) for x in brk]
        if isinstance(labels, Mapping):
            return [str(labels.get(x, x)) for x in brk]
        if self.breaks is None:
            if len(labels) != len(levels):
                raise ValueError(f"Got {len(labels)} labels for {len(levels)} levels")
            by_level = dict(zip(levels, labels))
        else:
            if callable(self.breaks):
                raise ValueError("a list of labels needs a list of breaks, not a callable")
            by_level = dict(zip((_to_python(b) for b in self.breaks), labels))
        return [str(by_level.get(x, x)) for x in brk]

    def get_breaks(self, limits=None) -> np.ndarray:
        """Float positions of the break levels, censored to numeric ``limits``."""
        return self._breaks_and_labels(limits)[0]

    def get_labels(self, limits=None) -> list[str]:
        """Labels for :meth:`get_breaks` (``limits`` as there)."""
        return self._breaks_and_labels(limits)[1]

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} aesthetics={self.aesthetics[0]!r} "
            f"levels={self.get_limits()!r} name={self.name!r}>"
        )


# --------------------------------------------------------------------------- constructors
@register
def scale_x_discrete(
    name=None, limits=None, breaks=None, labels=None, expand=None, drop=True, aesthetics=None
) -> ScaleDiscretePosition:
    """Discrete x position scale. See :class:`ScaleDiscretePosition` for arguments;
    ``aesthetics`` defaults to every x-like aesthetic."""
    return ScaleDiscretePosition(
        X_AESTHETICS if aesthetics is None else aesthetics,
        name=name,
        limits=limits,
        breaks=breaks,
        labels=labels,
        expand=expand,
        drop=drop,
    )


@register
def scale_y_discrete(
    name=None, limits=None, breaks=None, labels=None, expand=None, drop=True, aesthetics=None
) -> ScaleDiscretePosition:
    """Discrete y position scale. See :class:`ScaleDiscretePosition` for arguments;
    ``aesthetics`` defaults to every y-like aesthetic."""
    return ScaleDiscretePosition(
        Y_AESTHETICS if aesthetics is None else aesthetics,
        name=name,
        limits=limits,
        breaks=breaks,
        labels=labels,
        expand=expand,
        drop=drop,
    )
