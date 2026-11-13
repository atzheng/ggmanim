"""Colour-steps guide for binned colour / fill scales (ggplot2's ``guide_coloursteps``).

A :class:`Coloursteps` is a :class:`~ggmanim.guides_colourbar.Colourbar` whose
bar is one filled :class:`~manim.Rectangle` per bin (exact step edges, no
sampling), with labels at the bin boundaries (the scale's breaks strictly inside
its limits; also at the limits with ``show_limits=True``). With
``even_steps=True`` (the default, as in ggplot2) every bin gets the same length
regardless of its width in data; ``even_steps=False`` sizes the steps
proportionally to the bins. No tick marks by default.

It is the default guide of :class:`~ggmanim.scales.brewer.ScaleBinnedColour`
(``scale_*_fermenter``) and can be asked for on any continuous colour scale
(``guides(color="coloursteps")`` / ``guide_coloursteps()``), which is then
shown binned at its breaks, each step coloured at its bin's midpoint.
:func:`guide_colorsteps` is an alias of :func:`guide_coloursteps`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from manim import Mobject

from .guides_colourbar import Colourbar
from .labels import render_label
from .theme import Theme

__all__ = [
    "COLOURSTEPS_NAMES",
    "Coloursteps",
    "GuideColoursteps",
    "build_coloursteps",
    "guide_colorsteps",
    "guide_coloursteps",
    "is_coloursteps_spec",
]

#: Strings that select the colour-steps guide.
COLOURSTEPS_NAMES = ("coloursteps", "colorsteps")


class GuideColoursteps:
    """Colour-steps options (see :func:`guide_coloursteps`)."""

    __slots__ = (
        "title",
        "barwidth",
        "barheight",
        "even_steps",
        "show_limits",
        "reverse",
        "ticks",
        "order",
        "direction",
    )

    def __init__(
        self,
        title: Any = None,
        barwidth: float | None = None,
        barheight: float | None = None,
        even_steps: bool = True,
        show_limits: bool = False,
        reverse: bool = False,
        ticks: bool = False,
        order: int = 0,
        direction: str | None = None,
    ) -> None:
        if direction not in (None, "horizontal", "vertical"):
            raise ValueError(
                f"direction must be 'horizontal', 'vertical' or None, got {direction!r}"
            )
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise ValueError(f"order must be an integer >= 0, got {order!r}")
        self.title = title
        self.barwidth = barwidth
        self.barheight = barheight
        self.even_steps = bool(even_steps)
        self.show_limits = bool(show_limits)
        self.reverse = bool(reverse)
        self.ticks = bool(ticks)
        self.order = int(order)
        self.direction = direction

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GuideColoursteps) and all(
            getattr(self, s) == getattr(other, s) for s in self.__slots__
        )

    def __hash__(self) -> int:
        return hash(tuple(getattr(self, s) for s in self.__slots__))

    def __repr__(self) -> str:
        args = ", ".join(f"{s}={getattr(self, s)!r}" for s in self.__slots__)
        return f"guide_coloursteps({args})"


def guide_coloursteps(
    title: Any = None,
    barwidth: float | None = None,
    barheight: float | None = None,
    even_steps: bool = True,
    show_limits: bool = False,
    reverse: bool = False,
    ticks: bool = False,
    order: int = 0,
    direction: str | None = None,
) -> GuideColoursteps:
    """Colour-steps guide options: one coloured step per bin of a binned scale.

    ``even_steps=True`` draws equal-length steps whatever the bin widths
    (``False``: lengths proportional to the bins); ``show_limits=True`` also
    labels the two ends of the bar with the scale limits; ``ticks=True`` adds
    tick marks at the labelled boundaries. ``title``, ``barwidth``,
    ``barheight``, ``reverse``, ``order`` and ``direction`` work as in
    :func:`~ggmanim.guides_colourbar.guide_colorbar`.
    """
    return GuideColoursteps(
        title, barwidth, barheight, even_steps, show_limits, reverse, ticks, order, direction
    )


guide_colorsteps = guide_coloursteps


def is_coloursteps_spec(spec: Any) -> bool:
    """True if a guide spec (string or object) asks for colour steps."""
    return isinstance(spec, GuideColoursteps) or (
        isinstance(spec, str) and spec.lower() in COLOURSTEPS_NAMES
    )


class Coloursteps(Colourbar):
    """Colour steps: ``VGroup(title?, bar, ticks, labels)``, one bar cell per bin.

    As :class:`~ggmanim.guides_colourbar.Colourbar`, plus

    edges : bin edges (transformed space), low to high; ``len(edges) == len(rects) + 1``.
    breaks : the labelled boundaries (inner edges, plus the limits with ``show_limits``).
    """

    options: GuideColoursteps  # type: ignore[assignment]

    def __init__(
        self,
        title: Mobject | None,
        colours: Sequence[Any],
        edges: Sequence[float],
        breaks: Sequence[float],
        labels: Sequence[Mobject],
        aesthetics: tuple[str, ...],
        options: GuideColoursteps | None = None,
        theme: Theme | None = None,
        direction: str = "vertical",
    ) -> None:
        if len(edges) != len(colours) + 1:
            raise ValueError("need one more edge than colours")
        options = options if options is not None else GuideColoursteps()
        # set before Colourbar.__init__, which lays the bar out
        self.edges = [float(e) for e in edges]
        n = len(colours)
        if options.even_steps:
            self._edge_fracs = [i / n for i in range(n + 1)] if n else [0.0]
        else:
            lo, hi = self.edges[0], self.edges[-1]
            span = hi - lo
            self._edge_fracs = [
                (e - lo) / span if span else i / max(n, 1) for i, e in enumerate(self.edges)
            ]
        limits = (self.edges[0], self.edges[-1])
        super().__init__(
            title,
            colours,
            breaks,
            labels,
            limits,
            aesthetics,
            options,  # type: ignore[arg-type]  # duck-types GuideColourbar
            theme,
            direction,
        )

    def _cell_spans(self) -> list[tuple[float, float]]:
        f = self._edge_fracs
        return [(f[i], f[i + 1]) for i in range(len(self.rects))]

    def _fraction(self, value: float) -> float:
        f = float(np.interp(value, self.edges, self._edge_fracs))
        return 1.0 - f if self.options.reverse else f


def _with_limit_labels(
    scale: Any, shown: list[tuple[float, str]], lo: float, hi: float
) -> list[tuple[float, str]]:
    """``shown`` with the limits ``lo``/``hi`` (transformed space) added at the
    ends. Default and function labels format limits and breaks together (so the
    precision matches); a fixed label list keeps its labels and formats the limits."""
    values = [lo, *(b for b, _ in shown), hi]
    data = np.asarray(scale.trans.inverse(np.array(values, dtype=float)), dtype=float)
    labels = getattr(scale, "labels", None)
    if labels is None:
        texts = [str(t) for t in scale.trans.format(data)]
    elif callable(labels):
        texts = [str(t) for t in labels(data)]
    else:
        ends = [str(t) for t in scale.trans.format(data[[0, -1]])]
        texts = [ends[0], *(lab for _, lab in shown), ends[1]]
    return list(zip(values, texts))


def build_coloursteps(
    title: Any,
    aes_scales: dict[str, Any],
    breaks: Sequence[float],
    labels: Sequence[str],
    theme: Theme,
    options: GuideColoursteps | None = None,
) -> Coloursteps:
    """A :class:`Coloursteps` for the first scale in ``aes_scales``; ``breaks``
    (transformed) and ``labels`` are the scale's, the inner ones become the bin
    boundaries."""
    options = options if options is not None else GuideColoursteps()
    scale = next(iter(aes_scales.values()))
    limits = scale.get_limits()
    lo, hi = limits if limits is not None else (0.0, 1.0)
    pairs = sorted(
        {float(b): str(lab) for b, lab in zip(breaks, labels) if lo < float(b) < hi}.items()
    )
    if hasattr(scale, "bin_edges"):
        edges = [float(e) for e in scale.bin_edges()]
        colours = list(scale.bin_colours())
    else:
        edges = [lo] + [b for b, _ in pairs] + [hi]
        mids = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2
        colours = list(scale.map_transformed(mids))
    if len(edges) < 2:
        edges = [lo, hi]
    colours = colours[: len(edges) - 1]
    shown = [(b, lab) for b, lab in pairs if edges[0] < b < edges[-1]]
    if options.show_limits:
        shown = _with_limit_labels(scale, shown, edges[0], edges[-1])
    label_mobs = [
        theme.make_text(render_label(lab, theme.text_engine), **theme.text_style("legend_text"))
        for _, lab in shown
    ]
    if options.title is not None:
        title = options.title
    if title is not None:
        if isinstance(title, Mobject):
            title = title.copy()
        else:
            title = theme.make_text(str(title), **theme.text_style("legend_title"))
    return Coloursteps(
        title,
        colours,
        edges,
        [b for b, _ in shown],
        label_mobs,
        tuple(aes_scales),
        options,
        theme,
    )
