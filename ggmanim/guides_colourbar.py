"""Colourbar guide for continuous colour / fill scales.

A :class:`Colourbar` is ``VGroup(title?, bar, ticks, labels)``: the bar is ``nbin``
thin filled :class:`~manim.Rectangle` s (robust across renderers -- no sheen
gradients) coloured by the scale over its limits, with tick marks and labels at
``scale.get_breaks()``. It is vertical (low values at the bottom) for right/left
legends and horizontal (low values on the left) for top/bottom legends; like
:class:`~ggmanim.guides.Legend` it implements ``arrange_rows(direction)`` so
:func:`~ggmanim.guides.place_legends` treats both uniformly.

:func:`guide_colorbar` (alias :func:`guide_colourbar`) configures it, either on
the scale (``scale_color_gradient(guide=guide_colorbar(nbin=50))``) or via
``guides(color=guide_colorbar(reverse=True))``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from manim import LEFT, UP, Line, Mobject, Rectangle, SurroundingRectangle, VGroup

from .geoms.base import KEY_SIZE
from .theme import Theme, theme_default

__all__ = [
    "COLOURBAR_NAMES",
    "Colourbar",
    "GuideColourbar",
    "build_colourbar",
    "guide_colorbar",
    "guide_colourbar",
    "is_colourbar_spec",
]

#: Strings that select the colourbar guide.
COLOURBAR_NAMES = ("colourbar", "colorbar")

_LABEL_BUFF = 0.12
_TITLE_BUFF = 0.15


class GuideColourbar:
    """Colourbar options (see :func:`guide_colorbar`)."""

    __slots__ = ("title", "barwidth", "barheight", "nbin", "reverse", "ticks")

    def __init__(
        self,
        title: Any = None,
        barwidth: float | None = None,
        barheight: float | None = None,
        nbin: int = 32,
        reverse: bool = False,
        ticks: bool = True,
    ) -> None:
        if int(nbin) < 1:
            raise ValueError("nbin must be >= 1")
        self.title = title
        self.barwidth = barwidth
        self.barheight = barheight
        self.nbin = int(nbin)
        self.reverse = bool(reverse)
        self.ticks = bool(ticks)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GuideColourbar) and all(
            getattr(self, s) == getattr(other, s) for s in self.__slots__
        )

    def __hash__(self) -> int:
        return hash(tuple(getattr(self, s) for s in self.__slots__))

    def __repr__(self) -> str:
        args = ", ".join(f"{s}={getattr(self, s)!r}" for s in self.__slots__)
        return f"guide_colorbar({args})"


def guide_colorbar(
    title: Any = None,
    barwidth: float | None = None,
    barheight: float | None = None,
    nbin: int = 32,
    reverse: bool = False,
    ticks: bool = True,
) -> GuideColourbar:
    """Colourbar guide options.

    ``title`` overrides the scale/labs title; ``barwidth``/``barheight`` are the
    bar's thickness and length in scene units (default ``1.2 * KEY_SIZE`` and
    ``5`` times that); ``nbin`` is the number of rectangles the bar is made of;
    ``reverse`` puts high values at the bottom (left); ``ticks=False`` hides the
    tick marks.
    """
    return GuideColourbar(title, barwidth, barheight, nbin, reverse, ticks)


guide_colourbar = guide_colorbar


def is_colourbar_spec(spec: Any) -> bool:
    """True if a guide spec (string or object) asks for a colourbar."""
    return isinstance(spec, GuideColourbar) or (
        isinstance(spec, str) and spec.lower() in COLOURBAR_NAMES
    )


class Colourbar(VGroup):
    """A colourbar: ``VGroup(title?, bar, ticks, labels)``.

    Attributes
    ----------
    title : Mobject or None
    bar : VGroup of ``nbin`` filled Rectangles, low to high values.
    rects : list of Rectangle (the bar's cells, low to high).
    ticks : VGroup of tick-mark Lines (two per break, one on each long edge).
    labels : list of Mobject, one per break.
    colours : list of the ``nbin`` bar colours, low to high.
    breaks : break positions (the scale's transformed space).
    limits : ``(lo, hi)`` of the scale (transformed space).
    aesthetics : tuple of str, e.g. ``("color", "fill")`` when merged.
    direction : ``"vertical"`` or ``"horizontal"`` (current layout).
    """

    def __init__(
        self,
        title: Mobject | None,
        colours: Sequence[Any],
        breaks: Sequence[float],
        labels: Sequence[Mobject],
        limits: tuple[float, float],
        aesthetics: tuple[str, ...],
        options: GuideColourbar | None = None,
        theme: Theme | None = None,
        direction: str = "vertical",
    ) -> None:
        super().__init__()
        self.options = options if options is not None else GuideColourbar()
        self.theme = theme if theme is not None else theme_default()
        self.title = title
        self.colours = list(colours)
        self.breaks = [float(b) for b in breaks]
        self.labels = list(labels)
        self.limits = (float(limits[0]), float(limits[1]))
        self.aesthetics = tuple(aesthetics)
        self.rects: list[Rectangle] = [
            # a hairline stroke in the cell's own colour hides anti-aliasing seams
            Rectangle(width=1.0, height=1.0, fill_color=c, fill_opacity=1.0).set_stroke(
                color=c, width=0.5, opacity=1.0
            )
            for c in self.colours
        ]
        self.bar = VGroup(*self.rects)
        self.ticks = VGroup()
        self.box: Mobject | None = None
        self.direction = direction
        self.arrange_rows(direction)

    # -- geometry ------------------------------------------------------------
    @property
    def bar_thickness(self) -> float:
        bw = self.options.barwidth
        return float(bw) if bw is not None else KEY_SIZE * 1.2

    @property
    def bar_length(self) -> float:
        bh = self.options.barheight
        return float(bh) if bh is not None else KEY_SIZE * 1.2 * 5

    def _fraction(self, value: float) -> float:
        lo, hi = self.limits
        f = 0.5 if hi == lo else (value - lo) / (hi - lo)
        return 1.0 - f if self.options.reverse else f

    def arrange_rows(self, direction: str = "vertical") -> Colourbar:
        """(Re)lay out the bar vertically (low values at the bottom) or
        horizontally (low values on the left), with ticks, labels and title."""
        self.direction = "horizontal" if direction == "horizontal" else "vertical"
        horizontal = self.direction == "horizontal"
        thick, length = self.bar_thickness, self.bar_length
        n = max(len(self.rects), 1)
        cell = length / n
        for i, rect in enumerate(self.rects):
            pos = (i + 0.5) / n
            if self.options.reverse:
                pos = 1.0 - pos
            if horizontal:
                rect.stretch_to_fit_width(cell).stretch_to_fit_height(thick)
                rect.move_to((pos * length, 0.0, 0.0))
            else:
                rect.stretch_to_fit_width(thick).stretch_to_fit_height(cell)
                rect.move_to((0.0, pos * length, 0.0))

        tick_len = thick / 5
        ticks = []
        if self.options.ticks:
            for b in self.breaks:
                p = self._fraction(b) * length
                if horizontal:
                    ticks += [
                        Line((p, -thick / 2, 0), (p, -thick / 2 + tick_len, 0)),
                        Line((p, thick / 2, 0), (p, thick / 2 - tick_len, 0)),
                    ]
                else:
                    ticks += [
                        Line((-thick / 2, p, 0), (-thick / 2 + tick_len, p, 0)),
                        Line((thick / 2, p, 0), (thick / 2 - tick_len, p, 0)),
                    ]
        for t in ticks:
            t.set_stroke(color="#FFFFFF", width=1.5)
        self.ticks = VGroup(*ticks)

        for b, lab in zip(self.breaks, self.labels):
            p = self._fraction(b) * length
            if horizontal:
                lab.move_to((p, 0.0, 0.0))
                lab.align_to((0.0, -thick / 2 - _LABEL_BUFF, 0.0), UP)
            else:
                lab.move_to((0.0, p, 0.0))
                lab.align_to((thick / 2 + _LABEL_BUFF, 0.0, 0.0), LEFT)

        if self.title is not None:
            self.title.next_to(self.bar, UP, buff=_TITLE_BUFF, aligned_edge=LEFT)

        parts: list[Any] = [self.title] if self.title is not None else []
        parts += [self.bar, self.ticks, *self.labels]
        self.box = None
        if getattr(self.theme, "legend_box", False):
            self.box = SurroundingRectangle(
                VGroup(*parts), color=self.theme.get_foreground(), buff=0.15, stroke_width=1
            )
            parts.append(self.box)
        self.submobjects = []
        self.add(*parts)
        return self


def build_colourbar(
    title: Any,
    aes_scales: dict[str, Any],
    breaks: Sequence[float],
    labels: Sequence[str],
    theme: Theme,
    options: GuideColourbar | None = None,
) -> Colourbar:
    """A :class:`Colourbar` for the first scale in ``aes_scales`` (all of which
    share title, breaks and labels)."""
    options = options if options is not None else GuideColourbar()
    scale = next(iter(aes_scales.values()))
    limits = scale.get_limits()
    lo, hi = limits if limits is not None else (0.0, 1.0)
    n = options.nbin
    tv = lo + (np.arange(n) + 0.5) / n * (hi - lo)
    colours = list(scale.map_transformed(tv))
    label_mobs = [theme.make_text(str(lab), font_size=theme.legend_text_size) for lab in labels]
    if options.title is not None:
        title = options.title
    if title is not None:
        if isinstance(title, Mobject):
            title = title.copy()
        else:
            title = theme.make_text(str(title), font_size=theme.legend_title_size)
    return Colourbar(
        title, colours, breaks, label_mobs, (lo, hi), tuple(aes_scales), options, theme
    )
