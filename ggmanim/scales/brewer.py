"""ColorBrewer colour / fill scales: ``brewer`` (discrete), ``distiller``
(continuous) and ``fermenter`` (binned), after ggplot2.

* :func:`brewer_pal` -- ``scales::brewer_pal``: ``n -> n colours`` from the
  ColorBrewer scheme designed for ``n`` classes (tables in :mod:`ggmanim._brewer`).
  ``palette`` is a name (``"Set1"``, case-insensitive) or a 1-based index into the
  palettes of ``type`` (``"seq"``, ``"div"`` or ``"qual"``, ggplot2's order);
  ``direction=-1`` reverses the colours. As in ``scales``, ``n < 3`` takes the
  first ``n`` colours of the 3-class scheme. More classes than the palette defines
  raise a ``ValueError`` (ggplot2 warns and returns ``NA``\\s).
* ``scale_color_brewer`` -- a :class:`~.discrete.ScaleDiscrete` on ``brewer_pal``.
  ``cycle=True`` recycles the largest scheme instead of raising.
* ``scale_color_distiller`` -- a :class:`~.colour.ScaleContinuousColour`
  interpolating the 7-class scheme (Oklab), ``direction=-1`` by default as in
  ggplot2 (so for sequential palettes the darkest colour is at the low end).
* ``scale_color_fermenter`` -- :class:`ScaleBinnedColour`: the limits are cut at
  the scale's breaks into bins, bin ``i`` getting colour ``i`` of
  ``brewer_pal(n_bins)`` (``direction=-1`` by default). Bins are right-closed
  like ggplot2's (a value equal to a break falls in the lower bin). The guide is
  a colourbar finely sampled enough to show the steps.

All have ``fill`` and ``colour`` variants. No manim imports beyond the palettes.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from manim import ManimColor

from .._brewer import BREWER, BREWER_TYPES, brewer_max_n, brewer_type_of
from ..palettes import as_manim_color, gradient_n_pal
from ..registry import register
from .colour import ScaleContinuousColour
from .continuous import ScaleContinuous
from .discrete import ScaleDiscrete, _alias

__all__ = [
    "ScaleBinnedColour",
    "brewer_pal",
    "brewer_palette_name",
    "scale_color_brewer",
    "scale_color_distiller",
    "scale_color_fermenter",
    "scale_colour_brewer",
    "scale_colour_distiller",
    "scale_colour_fermenter",
    "scale_fill_brewer",
    "scale_fill_distiller",
    "scale_fill_fermenter",
]

_TYPE_ALIASES = {
    "seq": "seq",
    "sequential": "seq",
    "div": "div",
    "diverging": "div",
    "qual": "qual",
    "qualitative": "qual",
}
_BY_LOWER = {name.lower(): name for name in BREWER}
#: Number of colours of the scheme interpolated by ``distiller`` (as ggplot2).
_DISTILLER_N = 7
#: Colourbar resolution used by ``fermenter`` so the bin steps land close to the breaks.
_FERMENTER_NBIN = 100


def brewer_palette_name(palette: str | int = 1, type: str = "seq") -> str:
    """Resolve a palette name (case-insensitive) or 1-based index within ``type``
    to the canonical ColorBrewer name."""
    if isinstance(palette, str):
        try:
            return _BY_LOWER[palette.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown ColorBrewer palette {palette!r}; available: "
                + ", ".join(f"{t}: {', '.join(n)}" for t, n in BREWER_TYPES.items())
            ) from None
    try:
        t = _TYPE_ALIASES[type.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"type must be 'seq', 'div' or 'qual', got {type!r}") from None
    names = BREWER_TYPES[t]
    if isinstance(palette, bool) or not isinstance(palette, (int, np.integer)):
        raise TypeError(f"palette must be a name or an integer index, got {palette!r}")
    if not 1 <= palette <= len(names):
        raise ValueError(f"palette index {palette} out of range for type {t!r} (1..{len(names)})")
    return names[int(palette) - 1]


def _check_direction(direction: int) -> None:
    if direction not in (1, -1):
        raise ValueError("direction must be 1 or -1")


def _too_many(name: str, n: int) -> ValueError:
    mx = brewer_max_n(name)
    t = brewer_type_of(name)
    hint = (
        "use cycle=True, a larger palette (e.g. 'Paired' or 'Set3' have 12), "
        "scale_*_manual() or scale_*_hue()"
        if t == "qual"
        else "use scale_*_distiller() (continuous) or scale_*_fermenter() (binned) instead"
    )
    return ValueError(f"ColorBrewer palette {name!r} has at most {mx} colours, {n} needed: {hint}.")


def brewer_pal(
    type: str = "seq", palette: str | int = 1, direction: int = 1, *, strict: bool = True
) -> Callable[[int], list[ManimColor]]:
    """Discrete ColorBrewer palette ``n -> n colours`` (``scales::brewer_pal``).

    ``strict=False`` returns the largest scheme (fewer than ``n`` colours) instead of
    raising when ``n`` exceeds it, so callers can recycle.
    """
    _check_direction(direction)
    name = brewer_palette_name(palette, type)
    table = BREWER[name]
    mx = max(table)

    def pal(n: int) -> list[ManimColor]:
        if n <= 0:
            return []
        if n > mx:
            if strict:
                raise _too_many(name, n)
            cols = list(table[mx])
        else:
            cols = list(table[max(n, 3)])[:n]
        if direction == -1:
            cols.reverse()
        return [ManimColor(c) for c in cols]

    pal.__name__ = f"brewer_pal_{name}"
    return pal


# ---------------------------------------------------------------------------
# binned colour scale (fermenter)
# ---------------------------------------------------------------------------


class ScaleBinnedColour(ScaleContinuousColour):
    """Continuous colour scale cut into bins at its breaks, one colour per bin.

    ``bin_palette`` is a discrete palette ``n_bins -> colours`` (low to high).
    Bin edges are the limits plus the breaks strictly inside them (transformed
    space); bins are right-closed, the first one closed on both sides. Other
    arguments as for :class:`~.colour.ScaleContinuousColour`.
    """

    def __init__(self, aesthetics: Any, bin_palette: Callable[[int], list[Any]], **kwargs):
        if "guide" not in kwargs:
            from ..guides_colourbar import guide_colorbar  # lazy: keeps scales manim-free

            kwargs["guide"] = guide_colorbar(nbin=_FERMENTER_NBIN)
        super().__init__(aesthetics, None, **kwargs)
        self.bin_palette = bin_palette

    def bin_edges(self) -> np.ndarray:
        """Bin edges in transformed space (empty if the scale is untrained)."""
        limits = self.get_limits()
        if limits is None:
            return np.array([], dtype=float)
        lo, hi = limits
        br = np.asarray(self.get_breaks(), dtype=float)
        inner = np.unique(br[(br > lo) & (br < hi)])
        return np.concatenate([[lo], inner, [hi]])

    def bin_colours(self) -> list[Any]:
        """One colour per bin, low to high."""
        edges = self.bin_edges()
        return list(self.bin_palette(max(len(edges) - 1, 1))) if edges.size else []

    def rescale(self, values: Any) -> np.ndarray:
        """Bin midpoints rescaled to ``[0, 1]`` (NaN where missing / censored)."""
        x = np.asarray(ScaleContinuous.map_transformed(self, values), dtype=float)
        edges = self.bin_edges()
        if edges.size == 0:
            return np.full(x.shape, np.nan)
        nb = len(edges) - 1
        idx = np.clip(np.searchsorted(edges[1:-1], x, side="left"), 0, nb - 1)
        return np.where(np.isfinite(x), (idx + 0.5) / nb, np.nan)

    def map_transformed(self, values: Any) -> np.ndarray:  # type: ignore[override]
        x = np.asarray(ScaleContinuous.map_transformed(self, values), dtype=float).ravel()
        edges = self.bin_edges()
        out = np.empty(x.size, dtype=object)
        if edges.size == 0:
            out[:] = self.na_value
            return out
        colours = [as_manim_color(c) for c in self.bin_colours()]
        idx = np.clip(np.searchsorted(edges[1:-1], x, side="left"), 0, len(colours) - 1)
        for i, (v, k) in enumerate(zip(x, idx)):
            out[i] = colours[k] if np.isfinite(v) else self.na_value
        return out


# ---------------------------------------------------------------------------
# constructors
# ---------------------------------------------------------------------------


def _brewer(aesthetics, type, palette, direction, kwargs) -> ScaleDiscrete:
    pal = brewer_pal(type, palette, direction, strict=not kwargs.get("cycle", False))
    return ScaleDiscrete(aesthetics, palette=pal, **kwargs)


@register
def scale_color_brewer(
    type: str = "seq", palette: str | int = 1, direction: int = 1, *, aesthetics="color", **kwargs
) -> ScaleDiscrete:
    """Discrete ColorBrewer colour scale. ``palette``: name (``"Set1"``) or 1-based
    index into ``type`` (``"seq"``, ``"div"``, ``"qual"``); ``direction=-1`` reverses."""
    return _brewer(aesthetics, type, palette, direction, kwargs)


@register
def scale_fill_brewer(
    type: str = "seq", palette: str | int = 1, direction: int = 1, *, aesthetics="fill", **kwargs
) -> ScaleDiscrete:
    """Discrete ColorBrewer fill scale (see :func:`scale_color_brewer`)."""
    return _brewer(aesthetics, type, palette, direction, kwargs)


def _warn_qual(name: str, what: str) -> None:
    if brewer_type_of(name) == "qual":
        warnings.warn(
            f"Using a qualitative ColorBrewer palette ({name!r}) with a {what} scale; "
            "consider a sequential or diverging one.",
            stacklevel=3,
        )


def _distiller(aesthetics, type, palette, direction, values, kwargs) -> ScaleContinuousColour:
    _check_direction(direction)
    _warn_qual(brewer_palette_name(palette, type), "continuous")
    colours = brewer_pal(type, palette, direction)(_DISTILLER_N)
    return ScaleContinuousColour(aesthetics, gradient_n_pal(colours, values), **kwargs)


@register
def scale_color_distiller(
    type: str = "seq",
    palette: str | int = 1,
    direction: int = -1,
    values: Any = None,
    *,
    aesthetics="color",
    **kwargs,
) -> ScaleContinuousColour:
    """Continuous colour scale interpolating a 7-class ColorBrewer scheme.
    ``direction=-1`` (default, as ggplot2) puts the scheme's last colour at the low end."""
    return _distiller(aesthetics, type, palette, direction, values, kwargs)


@register
def scale_fill_distiller(
    type: str = "seq",
    palette: str | int = 1,
    direction: int = -1,
    values: Any = None,
    *,
    aesthetics="fill",
    **kwargs,
) -> ScaleContinuousColour:
    """Continuous ColorBrewer fill scale (see :func:`scale_color_distiller`)."""
    return _distiller(aesthetics, type, palette, direction, values, kwargs)


def _fermenter(aesthetics, type, palette, direction, kwargs) -> ScaleBinnedColour:
    _check_direction(direction)
    _warn_qual(brewer_palette_name(palette, type), "binned")
    return ScaleBinnedColour(aesthetics, brewer_pal(type, palette, direction), **kwargs)


@register
def scale_color_fermenter(
    type: str = "seq", palette: str | int = 1, direction: int = -1, *, aesthetics="color", **kwargs
) -> ScaleBinnedColour:
    """Binned ColorBrewer colour scale: the limits are cut at the breaks and each
    bin gets one colour of the scheme with that many classes."""
    return _fermenter(aesthetics, type, palette, direction, kwargs)


@register
def scale_fill_fermenter(
    type: str = "seq", palette: str | int = 1, direction: int = -1, *, aesthetics="fill", **kwargs
) -> ScaleBinnedColour:
    """Binned ColorBrewer fill scale (see :func:`scale_color_fermenter`)."""
    return _fermenter(aesthetics, type, palette, direction, kwargs)


scale_colour_brewer = _alias(scale_color_brewer, "scale_colour_brewer")
scale_colour_distiller = _alias(scale_color_distiller, "scale_colour_distiller")
scale_colour_fermenter = _alias(scale_color_fermenter, "scale_colour_fermenter")
