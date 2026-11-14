"""Strip mobjects: the labelled bars beside facet panels.

:func:`make_strip` draws one strip (background rectangle + centred text) of a
given length and thickness for a panel side; :func:`strip_thickness` picks the
common thickness of every strip in a plot (as ggplot2, all strips on one plot
have the same height so panels line up); :func:`attach_strips` stacks a panel's
strips on its sides.

Theme fields (region ``facets`` of :class:`~ggmanim.theme.Theme`):
``strip_background`` (fill colour; ``None`` = the foreground colour at
``strip_background_opacity``; ``False`` = no rectangle), ``strip_text_size``,
``strip_text_color`` (``None`` = foreground), ``strip_margin`` (padding round the
text).
"""

from __future__ import annotations

from collections.abc import Sequence

from manim import DL, PI, UR, ManimColor, Mobject, Rectangle, VGroup

from ..theme import Theme

__all__: list[str] = []

SIDES = ("top", "bottom", "left", "right")


def strip_text(theme: Theme, text: str, side: str) -> Mobject:
    """The (rotated for left/right strips) text mobject of one strip."""
    color = getattr(theme, "strip_text_color", None)
    style = theme.text_style("axis_text")
    style["font_size"] = getattr(theme, "strip_text_size", style["font_size"])
    style["color"] = ManimColor(color) if color is not None else theme.get_foreground()
    style.pop("align", None)
    mob = theme.make_text(str(text) if str(text) != "" else " ", **style)
    if side == "right":
        mob.rotate(-PI / 2)
    elif side == "left":
        mob.rotate(PI / 2)
    return mob


def strip_thickness(theme: Theme, labels: Sequence[str]) -> float:
    """Common strip thickness: the tallest strip text (``"Ag"`` at least, so
    strips do not depend on which letters a label has) plus twice ``strip_margin``."""
    probe = strip_text(theme, "Ag", "top").height
    tallest = max([probe, *(strip_text(theme, s, "top").height for s in labels)])
    return float(tallest + 2 * getattr(theme, "strip_margin", 0.08))


def make_strip(text: str, side: str, length: float, thickness: float, theme: Theme) -> VGroup:
    """A strip for ``side`` of a panel: a ``length`` x ``thickness`` rectangle
    (``thickness`` x ``length`` for left/right) with ``text`` centred in it, at
    the origin. ``strip.background`` (or ``None``) and ``strip.text`` are set."""
    if side not in SIDES:
        raise ValueError(f"strip side must be one of {SIDES}, got {side!r}")
    w, h = (length, thickness) if side in ("top", "bottom") else (thickness, length)
    rect = Rectangle(width=w, height=h, stroke_width=0)
    bg = getattr(theme, "strip_background", None)
    if bg is False:
        rect.set_fill(opacity=0.0)
        background = None
    else:
        color = ManimColor(bg) if bg is not None else theme.get_foreground()
        rect.set_fill(color, opacity=getattr(theme, "strip_background_opacity", 0.15))
        background = rect
    label = strip_text(theme, text, side)
    room = max(length - 2 * getattr(theme, "strip_margin", 0.08), 1e-3)
    along = label.width if side in ("top", "bottom") else label.height
    if along > room:  # shrink labels longer than the strip
        label.scale(room / along)
    label.move_to(rect.get_center())
    strip = VGroup(rect, label)  # type: ignore[arg-type]
    strip.background = background  # type: ignore[attr-defined]
    strip.text = label  # type: ignore[attr-defined]
    strip.side = side  # type: ignore[attr-defined]
    return strip


def attach_strips(
    box: Mobject,
    specs: Sequence[tuple[str, Sequence[str]]],
    thickness: float,
    theme: Theme,
    offsets: dict[str, float] | None = None,
) -> VGroup:
    """Strips for one panel, placed against ``box`` (the panel's plotting area).

    ``specs`` is ``[(side, labels), ...]``; ``labels[0]`` is the outermost strip
    on that side, the last one touches the panel (``offsets[side]`` further out,
    e.g. past an axis). Returns a ``VGroup`` of the strips.
    """
    lo, hi = box.get_corner(DL), box.get_corner(UR)
    out = VGroup()
    for side, labels in specs:
        start = (offsets or {}).get(side, 0.0)
        length = hi[0] - lo[0] if side in ("top", "bottom") else hi[1] - lo[1]
        for k, text in enumerate(reversed(list(labels))):
            strip = make_strip(text, side, length, thickness, theme)
            d = start + (k + 0.5) * thickness
            centre = (lo + hi) / 2
            if side == "top":
                centre[1] = hi[1] + d
            elif side == "bottom":
                centre[1] = lo[1] - d
            elif side == "left":
                centre[0] = lo[0] - d
            else:
                centre[0] = hi[0] + d
            strip.move_to(centre)
            out.add(strip)
    return out
