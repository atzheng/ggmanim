"""Range scales: continuous (and ordinal) size / alpha / linewidth.

A *range* scale maps data values into a numeric output interval ``range=(lo, hi)``
(ggplot2's ``scale_size`` / ``scale_alpha`` / ``scale_linewidth`` family):

* :class:`ScaleContinuousRange` -- a :class:`~.continuous.ScaleContinuous` whose
  :meth:`~ScaleContinuousRange.map` rescales *transformed* values from the scale
  limits to ``[0, 1]`` and then into ``range``, either linearly (``palette="linear"``,
  ``scales::rescale_pal``) or area-proportionally (``palette="area"``,
  ``scales::area_pal``: ``lo + (hi - lo) * sqrt(r)``).
  ``scale_size_area`` instead rescales by the largest absolute limit
  (``scales::abs_area``) so that 0 maps to size 0.
* :class:`ScaleContinuousIdentity` -- data are already output values (no legend).
* :func:`discrete_range_scale` -- a :class:`~.discrete.ScaleDiscrete` mapping the
  levels of an (ideally ordered) factor to evenly spaced values in ``range``.

Output units: ``size`` is the point radius in scene units, ``alpha`` an opacity in
``[0, 1]``, ``linewidth`` a manim ``stroke_width`` -- exactly what the geoms already
consume, so geoms are unchanged. Range scales don't expand their limits.

No manim imports here.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..registry import register
from .continuous import ScaleContinuous, Transform, _as_float_array
from .discrete import ScaleDiscrete

__all__ = [
    "RANGE_AESTHETICS",
    "DEFAULT_RANGES",
    "ScaleContinuousRange",
    "ScaleContinuousIdentity",
    "default_range_scale",
    "discrete_range_scale",
    "scale_alpha",
    "scale_alpha_identity",
    "scale_linewidth",
    "scale_linewidth_identity",
    "scale_radius",
    "scale_size",
    "scale_size_area",
    "scale_size_identity",
]

#: Aesthetics that get a range scale by default.
RANGE_AESTHETICS: tuple[str, ...] = ("size", "alpha", "linewidth")

#: Default output ranges (scene-unit radius, opacity, manim stroke width).
DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "size": (0.03, 0.15),
    "alpha": (0.1, 1.0),
    "linewidth": (1.0, 6.0),
}

_PALETTES = ("linear", "area")
_RESCALERS = ("range", "max")


class ScaleContinuousRange(ScaleContinuous):
    """Continuous scale mapping data into the numeric interval ``range``.

    Parameters
    ----------
    aesthetics : str or sequence of str
    range : ``(lo, hi)`` output interval.
    palette : ``"linear"`` (default) or ``"area"`` (``lo + (hi - lo) * sqrt(r)``).
    rescale : ``"range"`` (default: ``r = (t - min) / (max - min)`` over the limits)
        or ``"max"`` (``r = |t| / max |limits|``; with ``palette="area"`` and
        ``range=(0, max_size)`` this is :func:`scale_size_area`).
    na_value : output for missing / out-of-bounds values (default NaN: the geom
        falls back to its default).
    guide : ``"legend"`` (default) or ``"none"``.

    The remaining arguments (``name``, ``breaks``, ``labels``, ``limits``, ``oob``,
    ``trans``, ``n_breaks``) are as for :class:`~.continuous.ScaleContinuous`
    (``n_breaks`` defaults to 4, giving legends of about 3-5 keys); ``expand``
    is always ``False``.
    """

    def __init__(
        self,
        aesthetics,
        range: Sequence[float] = (0.0, 1.0),
        *,
        palette: str = "linear",
        rescale: str = "range",
        name: str | None = None,
        breaks: Sequence[float] | Callable | None = None,
        labels: Sequence[str] | Callable | None = None,
        limits: Sequence[float | None] | None = None,
        oob: str | Callable = "censor",
        trans: str | Transform | None = "identity",
        n_breaks: int = 4,
        na_value: float = np.nan,
        guide: str = "legend",
    ) -> None:
        super().__init__(
            aesthetics,
            name=name,
            breaks=breaks,
            labels=labels,
            limits=limits,
            expand=False,
            oob=oob,
            trans=trans,
            n_breaks=n_breaks,
        )
        rng = tuple(float(v) for v in range)
        if len(rng) != 2:
            raise ValueError(f"range must be (lo, hi), got {range!r}")
        if palette not in _PALETTES:
            raise ValueError(f"palette must be one of {_PALETTES}, got {palette!r}")
        if rescale not in _RESCALERS:
            raise ValueError(f"rescale must be one of {_RESCALERS}, got {rescale!r}")
        self.output_range: tuple[float, float] = (rng[0], rng[1])
        self.palette = palette
        self.rescale = rescale
        self.na_value = float(na_value)
        self.guide = guide

    #: Most keys a default legend shows (fewer pretty breaks are asked for if needed).
    max_legend_keys = 5

    def _default_breaks(self, dlo: float, dhi: float, limits) -> np.ndarray:
        b = super()._default_breaks(dlo, dhi, limits)
        n = self.n_breaks
        while n > 2 and self._n_inside(b, limits) > self.max_legend_keys:
            n -= 1
            b = np.asarray(self.trans.breaks(dlo, dhi, n), dtype=float)
        return b

    def _n_inside(self, data_breaks: np.ndarray, limits) -> int:
        t = self.trans.transform(data_breaks)
        return int(np.sum(np.isfinite(t) & (t >= limits[0]) & (t <= limits[1])))

    def rescale_unit(self, transformed: Any) -> np.ndarray:
        """Transformed values (after ``oob``) -> ``[0, 1]`` relative to the limits."""
        t = _as_float_array(transformed)
        limits = self.get_limits()
        if limits is None:
            return np.full(t.shape, np.nan)
        lo, hi = limits
        with np.errstate(invalid="ignore", divide="ignore"):
            if self.rescale == "max":
                m = max(abs(lo), abs(hi))
                return np.abs(t) / m if m > 0 else np.where(np.isfinite(t), 0.0, np.nan)
            if hi == lo:  # scales::rescale on a zero-width range -> midpoint
                return np.where(np.isfinite(t), 0.5, np.nan)
            return (t - lo) / (hi - lo)

    def map_transformed(self, values: Any) -> np.ndarray:
        """Transformed values -> output in ``range`` (``na_value`` for NaN / oob)."""
        t = super().map_transformed(values)  # oob against the limits
        r = self.rescale_unit(t)
        lo, hi = self.output_range
        with np.errstate(invalid="ignore"):
            if self.palette == "area":
                r = np.sqrt(np.clip(r, 0.0, None))
            out = lo + (hi - lo) * r
        return np.where(np.isfinite(out), out, self.na_value)

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} aesthetics={self.aesthetics[0]!r} "
            f"range={self.output_range!r} palette={self.palette!r} limits={self.limits!r}>"
        )


class ScaleContinuousIdentity(ScaleContinuous):
    """Numeric data are already output values (e.g. radii in scene units). No legend
    by default."""

    def __init__(self, aesthetics, *, name: str | None = None, guide: str = "none") -> None:
        super().__init__(aesthetics, name=name, expand=False, oob="keep")
        self.guide = guide

    def map_transformed(self, values: Any) -> np.ndarray:
        return _as_float_array(values)


# --------------------------------------------------------------------------- discrete


def _linear_pal(rng: Sequence[float]) -> Callable[[int], list[float]]:
    lo, hi = (float(v) for v in rng)
    return lambda n: [float(v) for v in np.linspace(lo, hi, n)]


def _area_pal(rng: Sequence[float]) -> Callable[[int], list[float]]:
    lo, hi = (float(v) for v in rng)
    return lambda n: [float(v) for v in np.sqrt(np.linspace(lo**2, hi**2, n))]


def discrete_range_scale(aesthetic: str, range: Sequence[float] | None = None, **kwargs):
    """Discrete (ordinal) scale mapping levels to evenly spaced values in ``range``:
    linear for ``alpha`` / ``linewidth``, evenly spaced *areas* for ``size``
    (ggplot2's ``scale_size_ordinal``)."""
    rng = DEFAULT_RANGES[aesthetic] if range is None else range
    pal = _area_pal(rng) if aesthetic == "size" else _linear_pal(rng)
    return ScaleDiscrete(aesthetic, palette=pal, **kwargs)


def _is_ordered(values: Any) -> bool:
    dtype = getattr(values, "dtype", None)
    if isinstance(values, pd.Categorical):
        dtype = values.dtype
    return isinstance(dtype, pd.CategoricalDtype) and bool(dtype.ordered)


def default_range_scale(aesthetic: str, values: Any):
    """Default untrained scale for ``size`` / ``alpha`` / ``linewidth`` mapped to
    ``values``: a :class:`ScaleContinuousRange` for numeric data, else a discrete
    range scale (with a warning unless ``values`` is an ordered Categorical, as
    ggplot2 does)."""
    from .discrete import is_discrete

    if aesthetic not in RANGE_AESTHETICS:
        raise ValueError(f"no range scale for aesthetic {aesthetic!r}")
    if is_discrete(values):
        if not _is_ordered(values):
            warnings.warn(
                f"Using {aesthetic} for a discrete variable is not advised.",
                UserWarning,
                stacklevel=3,
            )
        return discrete_range_scale(aesthetic)
    return _CONSTRUCTORS[aesthetic]()


# --------------------------------------------------------------------------- constructors


@register
def scale_size(
    name=None,
    breaks=None,
    labels=None,
    limits=None,
    range=DEFAULT_RANGES["size"],
    trans="identity",
    oob="censor",
    n_breaks=4,
    guide="legend",
) -> ScaleContinuousRange:
    """Continuous point size (radius, scene units) scaled so *area* is proportional
    to the value: ``radius = lo + (hi - lo) * sqrt(r)`` with ``r`` in ``[0, 1]``."""
    return ScaleContinuousRange(
        "size",
        range,
        palette="area",
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        trans=trans,
        oob=oob,
        n_breaks=n_breaks,
        guide=guide,
    )


@register
def scale_size_area(
    name=None,
    breaks=None,
    labels=None,
    limits=None,
    max_size=DEFAULT_RANGES["size"][1],
    trans="identity",
    oob="censor",
    n_breaks=4,
    guide="legend",
) -> ScaleContinuousRange:
    """Size scale where 0 has radius 0 and area is proportional to ``|value|``;
    the largest absolute value gets radius ``max_size``."""
    return ScaleContinuousRange(
        "size",
        (0.0, max_size),
        palette="area",
        rescale="max",
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        trans=trans,
        oob=oob,
        n_breaks=n_breaks,
        guide=guide,
    )


@register
def scale_radius(
    name=None,
    breaks=None,
    labels=None,
    limits=None,
    range=DEFAULT_RANGES["size"],
    trans="identity",
    oob="censor",
    n_breaks=4,
    guide="legend",
) -> ScaleContinuousRange:
    """Size scale where the *radius* is linear in the value."""
    return ScaleContinuousRange(
        "size",
        range,
        palette="linear",
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        trans=trans,
        oob=oob,
        n_breaks=n_breaks,
        guide=guide,
    )


@register
def scale_alpha(
    name=None,
    breaks=None,
    labels=None,
    limits=None,
    range=DEFAULT_RANGES["alpha"],
    trans="identity",
    oob="censor",
    n_breaks=4,
    guide="legend",
) -> ScaleContinuousRange:
    """Continuous opacity, linear in the value, within ``range`` (default 0.1 .. 1)."""
    return ScaleContinuousRange(
        "alpha",
        range,
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        trans=trans,
        oob=oob,
        n_breaks=n_breaks,
        guide=guide,
    )


@register
def scale_linewidth(
    name=None,
    breaks=None,
    labels=None,
    limits=None,
    range=DEFAULT_RANGES["linewidth"],
    trans="identity",
    oob="censor",
    n_breaks=4,
    guide="legend",
) -> ScaleContinuousRange:
    """Continuous line width (manim ``stroke_width``), linear in the value."""
    return ScaleContinuousRange(
        "linewidth",
        range,
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        trans=trans,
        oob=oob,
        n_breaks=n_breaks,
        guide=guide,
    )


@register
def scale_size_identity(name=None, guide="none") -> ScaleContinuousIdentity:
    """Use numeric data directly as point radii (scene units). No legend by default."""
    return ScaleContinuousIdentity("size", name=name, guide=guide)


@register
def scale_alpha_identity(name=None, guide="none") -> ScaleContinuousIdentity:
    """Use numeric data directly as opacities. No legend by default."""
    return ScaleContinuousIdentity("alpha", name=name, guide=guide)


@register
def scale_linewidth_identity(name=None, guide="none") -> ScaleContinuousIdentity:
    """Use numeric data directly as stroke widths. No legend by default."""
    return ScaleContinuousIdentity("linewidth", name=name, guide=guide)


_CONSTRUCTORS: dict[str, Callable[[], ScaleContinuousRange]] = {
    "size": scale_size,
    "alpha": scale_alpha,
    "linewidth": scale_linewidth,
}
