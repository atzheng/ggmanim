"""Continuous colour / fill scales (gradients, viridis) and extra discrete colour scales.

:class:`ScaleContinuousColour` is a :class:`~.continuous.ScaleContinuous` whose
output is a colour: a data value is transformed (``trans``), handled by ``oob``
against the limits, rescaled to a position in ``[0, 1]`` and looked up in a
*continuous palette* (``t -> colours``, see :func:`ggmanim.palettes.gradient_n_pal`;
gradients are interpolated in Oklab space). Missing or out-of-bounds values map
to ``na_value`` (grey50 by default). Breaks/labels work exactly like a position
scale's and drive the colourbar guide (:mod:`ggmanim.guides_colourbar`).

Constructors (``color``/``fill``, with ``colour`` aliases)::

    scale_color_gradient(low, high)                 # ggplot2 default '#132B43' -> '#56B1F7'
    scale_color_gradient2(low, mid, high, midpoint) # diverging around ``midpoint``
    scale_color_gradientn(colours, values)
    scale_color_viridis_c(option, begin, end, direction)
    scale_color_continuous(type="gradient" | "viridis")
    scale_color_viridis_d(...) / scale_color_hue(h, c, l)   # discrete

No manim imports beyond ``ManimColor`` (via the palettes).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..palettes import (
    ContinuousPalette,
    as_manim_color,
    gradient_n_pal,
    hue_pal,
    viridis_pal,
)
from ..registry import register
from .continuous import ScaleContinuous
from .discrete import NA_COLOR, ScaleDiscrete, _alias, normalise_aesthetic

__all__ = [
    "ScaleContinuousColour",
    "rescale",
    "rescale_mid",
    "scale_color_continuous",
    "scale_color_gradient",
    "scale_color_gradient2",
    "scale_color_gradientn",
    "scale_color_hue",
    "scale_color_viridis_c",
    "scale_color_viridis_d",
    "scale_colour_continuous",
    "scale_colour_gradient",
    "scale_colour_gradient2",
    "scale_colour_gradientn",
    "scale_colour_hue",
    "scale_colour_viridis_c",
    "scale_colour_viridis_d",
    "scale_fill_continuous",
    "scale_fill_gradient",
    "scale_fill_gradient2",
    "scale_fill_gradientn",
    "scale_fill_hue",
    "scale_fill_viridis_c",
    "scale_fill_viridis_d",
]

#: ggplot2's default continuous colour gradient.
DEFAULT_LOW, DEFAULT_HIGH = "#132B43", "#56B1F7"
#: ggplot2's ``scale_colour_gradient2`` defaults (``muted("red")``, white, ``muted("blue")``).
DEFAULT_LOW2, DEFAULT_MID2, DEFAULT_HIGH2 = "#832424", "#FFFFFF", "#3A3A98"
#: Number of samples of a viridis map used as gradient stops by ``viridis_c``.
_VIRIDIS_STOPS = 64

Rescaler = Callable[[np.ndarray, tuple[float, float]], np.ndarray]


def rescale(x: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    """Linearly map ``limits`` to ``[0, 1]`` (a zero-width range maps to 0.5)."""
    x = np.asarray(x, dtype=float)
    lo, hi = limits
    if hi == lo:
        return np.where(np.isfinite(x), 0.5, np.nan)
    return (x - lo) / (hi - lo)


def rescale_mid(mid: float) -> Rescaler:
    """Rescaler putting ``mid`` (transformed space) at 0.5, symmetric about it
    (``scales::rescale_mid``): the farther limit lands on 0 or 1."""

    def f(x: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        extent = max(abs(limits[0] - mid), abs(limits[1] - mid))
        if extent == 0:
            return np.where(np.isfinite(x), 0.5, np.nan)
        return (x - mid) / (2 * extent) + 0.5

    return f


class ScaleContinuousColour(ScaleContinuous):
    """Continuous scale mapping numbers to colours through a continuous palette.

    Parameters
    ----------
    aesthetics : ``"color"``, ``"fill"`` or both.
    palette : ``t -> colours`` for positions ``t`` in ``[0, 1]`` (default: the
        ggplot2 gradient ``#132B43 -> #56B1F7``).
    na_value : colour for missing / out-of-bounds values (default grey50).
    guide : ``"colourbar"`` (default; also ``"colorbar"``), ``"legend"`` (one key
        per break), ``"none"``, or a :func:`~ggmanim.guides_colourbar.guide_colorbar`.
    rescaler : ``f(transformed_values, limits) -> [0, 1] positions`` (default
        linear, :func:`rescale`).
    name, breaks, labels, limits, oob, trans, n_breaks :
        As for :class:`~.continuous.ScaleContinuous` (``n_breaks`` defaults to 4
        so colourbar labels do not crowd). ``oob`` defaults to
        ``"censor"`` (out-of-range values get ``na_value``); ``"squish"`` maps them
        to the end colours.
    """

    is_colour = True

    def __init__(
        self,
        aesthetics: str | Sequence[str] = "color",
        palette: ContinuousPalette | None = None,
        *,
        name: str | None = None,
        breaks: Sequence[float] | Callable | None = None,
        labels: Sequence[str] | Callable | None = None,
        limits: Sequence[float | None] | None = None,
        oob: str | Callable = "censor",
        trans: Any = "identity",
        n_breaks: int = 4,
        na_value: Any = NA_COLOR,
        guide: Any = "colourbar",
        rescaler: Rescaler | None = None,
    ) -> None:
        if isinstance(aesthetics, str):
            aesthetics = (aesthetics,)
        super().__init__(
            tuple(normalise_aesthetic(a) for a in aesthetics),
            name=name,
            breaks=breaks,
            labels=labels,
            limits=limits,
            expand=False,
            oob=oob,
            trans=trans,
            n_breaks=n_breaks,
        )
        self.palette = (
            palette if palette is not None else gradient_n_pal([DEFAULT_LOW, DEFAULT_HIGH])
        )
        self.na_value = as_manim_color(na_value) if na_value is not None else None
        self.guide = guide
        self.rescaler: Rescaler = rescaler if rescaler is not None else rescale

    # ------------------------------------------------------------- mapping
    def rescale(self, values: Any) -> np.ndarray:
        """Transformed-space values -> palette positions in ``[0, 1]`` after ``oob``
        (NaN for missing / censored values)."""
        x = np.asarray(super().map_transformed(values), dtype=float)
        limits = self.get_limits()
        if limits is None:
            return np.full(x.shape, np.nan)
        return np.asarray(self.rescaler(x, limits), dtype=float)

    def map_transformed(self, values: Any) -> np.ndarray:  # type: ignore[override]
        """Transformed-space values -> object array of ``ManimColor`` (``na_value``
        where missing or out of bounds)."""
        t = self.rescale(values)
        out = np.empty(t.size, dtype=object)
        cols = self.palette(t.ravel())
        for i, c in enumerate(cols):
            out[i] = self.na_value if c is None else as_manim_color(c)
        return out

    def map(self, values: Any) -> Any:  # type: ignore[override]
        """Data values -> colours: a ``pd.Series`` (same index) for Series input,
        else a 1-d object ``np.ndarray``."""
        index = values.index if isinstance(values, pd.Series) else None
        out = self.map_transformed(self.transform(values))
        if index is not None:
            return pd.Series(out, index=index, dtype=object)
        return out

    def get_colours(self, n: int = 20) -> list[Any]:
        """``n`` colours evenly spaced over the limits (low to high), e.g. for a colourbar."""
        limits = self.get_limits()
        if limits is None or n <= 0:
            return []
        lo, hi = limits
        tv = lo + (np.arange(n) + 0.5) / n * (hi - lo)
        return list(self.map_transformed(tv))


# ---------------------------------------------------------------------------
# constructors
# ---------------------------------------------------------------------------


def _gradient(aesthetics, palette, **kwargs) -> ScaleContinuousColour:
    return ScaleContinuousColour(aesthetics, palette, **kwargs)


@register
def scale_color_gradient(
    low: Any = DEFAULT_LOW, high: Any = DEFAULT_HIGH, *, aesthetics="color", **kwargs
) -> ScaleContinuousColour:
    """Continuous two-colour gradient ``low -> high`` (ggplot2's default colour scale)."""
    return _gradient(aesthetics, gradient_n_pal([low, high]), **kwargs)


@register
def scale_fill_gradient(
    low: Any = DEFAULT_LOW, high: Any = DEFAULT_HIGH, *, aesthetics="fill", **kwargs
) -> ScaleContinuousColour:
    """Continuous two-colour fill gradient ``low -> high``."""
    return _gradient(aesthetics, gradient_n_pal([low, high]), **kwargs)


def _gradient2(aesthetics, low, mid, high, midpoint, kwargs) -> ScaleContinuousColour:
    scale = _gradient(aesthetics, gradient_n_pal([low, mid, high]), **kwargs)
    tmid = float(scale.trans.transform(np.array([midpoint], dtype=float))[0])
    scale.rescaler = rescale_mid(tmid)
    return scale


@register
def scale_color_gradient2(
    low: Any = DEFAULT_LOW2,
    mid: Any = DEFAULT_MID2,
    high: Any = DEFAULT_HIGH2,
    midpoint: float = 0.0,
    *,
    aesthetics="color",
    **kwargs,
) -> ScaleContinuousColour:
    """Diverging gradient ``low -> mid -> high`` with ``mid`` at data value ``midpoint``."""
    return _gradient2(aesthetics, low, mid, high, midpoint, kwargs)


@register
def scale_fill_gradient2(
    low: Any = DEFAULT_LOW2,
    mid: Any = DEFAULT_MID2,
    high: Any = DEFAULT_HIGH2,
    midpoint: float = 0.0,
    *,
    aesthetics="fill",
    **kwargs,
) -> ScaleContinuousColour:
    """Diverging fill gradient ``low -> mid -> high`` with ``mid`` at ``midpoint``."""
    return _gradient2(aesthetics, low, mid, high, midpoint, kwargs)


def _gradientn(aesthetics, colours, colors, values, kwargs) -> ScaleContinuousColour:
    cols = colours if colours is not None else colors
    if cols is None:
        raise TypeError("scale_*_gradientn() needs colours=[...]")
    return _gradient(aesthetics, gradient_n_pal(list(cols), values), **kwargs)


@register
def scale_color_gradientn(
    colours: Sequence[Any] | None = None,
    values: Sequence[float] | None = None,
    *,
    colors: Sequence[Any] | None = None,
    aesthetics="color",
    **kwargs,
) -> ScaleContinuousColour:
    """Gradient through ``colours`` placed at ``values`` in ``[0, 1]`` (default evenly)."""
    return _gradientn(aesthetics, colours, colors, values, kwargs)


@register
def scale_fill_gradientn(
    colours: Sequence[Any] | None = None,
    values: Sequence[float] | None = None,
    *,
    colors: Sequence[Any] | None = None,
    aesthetics="fill",
    **kwargs,
) -> ScaleContinuousColour:
    """Fill gradient through ``colours`` placed at ``values`` in ``[0, 1]``."""
    return _gradientn(aesthetics, colours, colors, values, kwargs)


def _viridis_c(aesthetics, option, begin, end, direction, kwargs) -> ScaleContinuousColour:
    stops = viridis_pal(begin, end, direction, option)(_VIRIDIS_STOPS)
    return _gradient(aesthetics, gradient_n_pal(stops), **kwargs)


@register
def scale_color_viridis_c(
    option: str = "viridis",
    begin: float = 0.0,
    end: float = 1.0,
    direction: int = 1,
    *,
    aesthetics="color",
    **kwargs,
) -> ScaleContinuousColour:
    """Continuous viridis-family colour scale. ``option``: ``"viridis"``, ``"magma"``,
    ``"inferno"``, ``"plasma"``, ``"cividis"`` (or ``"A"``..``"E"``)."""
    return _viridis_c(aesthetics, option, begin, end, direction, kwargs)


@register
def scale_fill_viridis_c(
    option: str = "viridis",
    begin: float = 0.0,
    end: float = 1.0,
    direction: int = 1,
    *,
    aesthetics="fill",
    **kwargs,
) -> ScaleContinuousColour:
    """Continuous viridis-family fill scale (see :func:`scale_color_viridis_c`)."""
    return _viridis_c(aesthetics, option, begin, end, direction, kwargs)


def _continuous(aesthetics, type, kwargs) -> ScaleContinuousColour:
    if type == "gradient":
        return _gradient(aesthetics, gradient_n_pal([DEFAULT_LOW, DEFAULT_HIGH]), **kwargs)
    if type == "viridis":
        return _viridis_c(aesthetics, "viridis", 0.0, 1.0, 1, kwargs)
    raise ValueError(f"type must be 'gradient' or 'viridis', got {type!r}")


@register
def scale_color_continuous(
    *, type: str = "gradient", aesthetics="color", **kwargs
) -> ScaleContinuousColour:
    """Default continuous colour scale: ``type="gradient"`` (default) or ``"viridis"``."""
    return _continuous(aesthetics, type, kwargs)


@register
def scale_fill_continuous(
    *, type: str = "gradient", aesthetics="fill", **kwargs
) -> ScaleContinuousColour:
    """Default continuous fill scale: ``type="gradient"`` (default) or ``"viridis"``."""
    return _continuous(aesthetics, type, kwargs)


@register
def scale_color_viridis_d(
    option: str = "viridis",
    begin: float = 0.0,
    end: float = 1.0,
    direction: int = 1,
    *,
    aesthetics="color",
    **kwargs,
) -> ScaleDiscrete:
    """Discrete colour scale sampling a viridis-family map evenly."""
    return ScaleDiscrete(aesthetics, palette=viridis_pal(begin, end, direction, option), **kwargs)


@register
def scale_fill_viridis_d(
    option: str = "viridis",
    begin: float = 0.0,
    end: float = 1.0,
    direction: int = 1,
    *,
    aesthetics="fill",
    **kwargs,
) -> ScaleDiscrete:
    """Discrete fill scale sampling a viridis-family map evenly."""
    return ScaleDiscrete(aesthetics, palette=viridis_pal(begin, end, direction, option), **kwargs)


@register
def scale_color_hue(
    h: tuple[float, float] = (15, 375),
    c: float = 100,
    l: float = 65,
    h_start: float = 0,
    direction: int = 1,
    *,
    aesthetics="color",
    **kwargs,
) -> ScaleDiscrete:
    """Discrete colour scale of evenly spaced HCL hues (ggplot2's default)."""
    return ScaleDiscrete(aesthetics, palette=hue_pal(h, c, l, h_start, direction), **kwargs)


@register
def scale_fill_hue(
    h: tuple[float, float] = (15, 375),
    c: float = 100,
    l: float = 65,
    h_start: float = 0,
    direction: int = 1,
    *,
    aesthetics="fill",
    **kwargs,
) -> ScaleDiscrete:
    """Discrete fill scale of evenly spaced HCL hues (ggplot2's default)."""
    return ScaleDiscrete(aesthetics, palette=hue_pal(h, c, l, h_start, direction), **kwargs)


scale_colour_gradient = _alias(scale_color_gradient, "scale_colour_gradient")
scale_colour_gradient2 = _alias(scale_color_gradient2, "scale_colour_gradient2")
scale_colour_gradientn = _alias(scale_color_gradientn, "scale_colour_gradientn")
scale_colour_viridis_c = _alias(scale_color_viridis_c, "scale_colour_viridis_c")
scale_colour_viridis_d = _alias(scale_color_viridis_d, "scale_colour_viridis_d")
scale_colour_continuous = _alias(scale_color_continuous, "scale_colour_continuous")
scale_colour_hue = _alias(scale_color_hue, "scale_colour_hue")
