"""Themes: the non-data appearance of a plot.

A :class:`Theme` is an immutable (frozen) dataclass. Complete themes
(:func:`theme_default`, :func:`theme_minimal`, :func:`theme_classic`) set
``complete=True`` and *replace* the plot's theme when added; :func:`theme`
builds a *partial* theme whose explicitly given fields are merged onto the
current theme (``old.merge(new)``, see :meth:`ggmanim.plot.ggplot._with_theme`).
A partial theme added to a plot without a theme is the default theme with the
overrides applied.

Nothing here reads manim's global ``config`` at import time: the foreground
colour (when ``foreground`` is ``None``) and the default panel size are resolved
when a plot is built, via :meth:`Theme.get_foreground` and :meth:`Theme.panel_size`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import manim
from manim import BLACK, WHITE, ManimColor, Mobject, config

from .registry import register

__all__ = [
    "Theme",
    "theme",
    "theme_default",
    "theme_minimal",
    "theme_classic",
    "foreground_for_background",
]

_TEXT_ENGINES = ("text", "tex")
_LEGEND_POSITIONS = ("right", "left", "top", "bottom", "none")


def foreground_for_background(background: Any) -> ManimColor:
    """``BLACK`` on a light background, ``WHITE`` on a dark one (relative luminance)."""
    r, g, b = ManimColor(background).to_rgb()[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return ManimColor(BLACK) if luminance > 0.5 else ManimColor(WHITE)


@dataclass(frozen=True)
class Theme:
    """Appearance settings for a plot.

    Text
        ``text_engine``: ``"text"`` (Pango :class:`~manim.Text`; no LaTeX needed) or
        ``"tex"`` (:class:`~manim.Tex`). ``font``: Pango font family (``None`` = manim
        default). ``axis_text_size`` (tick labels), ``axis_title_size``,
        ``legend_text_size``, ``legend_title_size``, ``title_size``.
    Colour
        ``foreground``: colour for axes, ticks and text; ``None`` derives it from
        ``config.background_color`` at build time (BLACK on light, WHITE on dark).
    Axes
        ``axis_line``: draw the axis lines; ``axis_stroke_width``; ``ticks``: draw tick
        marks; ``tick_length`` (scene units, drawn outward); ``tick_labels``: draw tick
        labels; ``tick_label_buff`` (gap between tick and label); ``axis_title_buff``.
    Grid
        ``panel_grid``: draw major grid lines at the breaks; ``grid_stroke_width``,
        ``grid_opacity``.
    Layout
        ``legend_position``: ``"right"`` (default), ``"left"``, ``"top"``,
        ``"bottom"`` or ``"none"``. ``panel_width``/``panel_height``: panel size in
        scene units; ``None`` fits manim's frame leaving ``legend_room`` for a legend
        (see :meth:`panel_size`).

    ``complete`` marks a complete theme (replaces the plot's theme when added);
    ``overrides`` lists the fields a partial :func:`theme` sets explicitly.
    """

    text_engine: str = "text"
    font: str | None = None
    axis_text_size: float = 24
    axis_title_size: float = 28
    legend_text_size: float = 24
    legend_title_size: float = 28
    title_size: float = 36

    foreground: Any | None = None

    axis_line: bool = True
    axis_stroke_width: float = 2.0
    ticks: bool = True
    tick_length: float = 0.12
    tick_labels: bool = True
    tick_label_buff: float = 0.1
    axis_title_buff: float = 0.25

    panel_grid: bool = True
    grid_stroke_width: float = 1.0
    grid_opacity: float = 0.15

    legend_position: str = "right"
    legend_room: float = 2.5
    panel_width: float | None = None
    panel_height: float | None = None

    complete: bool = False
    overrides: frozenset[str] = field(default=frozenset(), compare=False)

    def __post_init__(self):
        if self.text_engine not in _TEXT_ENGINES:
            raise ValueError(
                f"text_engine must be one of {_TEXT_ENGINES}, got {self.text_engine!r}"
            )
        if self.legend_position not in _LEGEND_POSITIONS:
            raise ValueError(
                f"legend_position must be one of {_LEGEND_POSITIONS}, got {self.legend_position!r}"
            )

    # -- composition ---------------------------------------------------------
    def add_to(self, plot):
        return plot._with_theme(self)

    def merge(self, other: Theme) -> Theme:
        """``self`` with ``other`` applied: a complete ``other`` replaces ``self``;
        a partial one overrides only its explicitly set fields."""
        if other.complete:
            return other
        changes = {name: getattr(other, name) for name in other.overrides}
        return dataclasses.replace(self, overrides=self.overrides | other.overrides, **changes)

    def replace(self, **changes) -> Theme:
        """A copy with ``changes`` applied (validated)."""
        return dataclasses.replace(self, **changes)

    # -- resolved values (read manim config at call time) --------------------
    def get_foreground(self) -> ManimColor:
        """The foreground colour; derived from ``config.background_color`` if unset."""
        if self.foreground is not None:
            return ManimColor(self.foreground)
        return foreground_for_background(config.background_color)

    def panel_size(self, has_legend: bool = False) -> tuple[float, float]:
        """``(width, height)`` of the panel (axes) in scene units.

        Unset dimensions fit the current manim frame with a 1-unit margin each side
        for tick labels / titles, minus ``legend_room`` on the legend's side when
        ``has_legend`` and the legend is shown.
        """
        legend = has_legend and self.legend_position != "none"
        w = self.panel_width
        if w is None:
            w = config.frame_width - 2.5
            if legend and self.legend_position in ("right", "left"):
                w -= self.legend_room
        h = self.panel_height
        if h is None:
            h = config.frame_height - 2.0
            if legend and self.legend_position in ("top", "bottom"):
                h -= min(self.legend_room, 1.0)
        return float(max(w, 0.5)), float(max(h, 0.5))

    # -- text factory --------------------------------------------------------
    def make_text(
        self, content: Any, font_size: float | None = None, color: Any = None, **kwargs
    ) -> Mobject:
        """Make a text mobject honouring ``text_engine``.

        A ready-made :class:`~manim.Mobject` is returned *unchanged* (the same object;
        callers that move or rotate it must ``.copy()`` it first). Anything else is
        converted with ``str``. ``font_size`` defaults to ``axis_text_size``; ``color``
        to :meth:`get_foreground`.
        """
        if isinstance(content, Mobject):
            return content
        text = str(content)
        if font_size is None:
            font_size = self.axis_text_size
        if color is None:
            color = self.get_foreground()
        if self.text_engine == "tex":
            return manim.Tex(text, font_size=font_size, color=color, **kwargs)
        if self.font is not None:
            kwargs.setdefault("font", self.font)
        return manim.Text(text, font_size=font_size, color=color, **kwargs)


_FIELD_NAMES = frozenset(f.name for f in dataclasses.fields(Theme)) - {"complete", "overrides"}


@register
def theme_default() -> Theme:
    """Default complete theme: axis lines, outward ticks, faint major grid."""
    return Theme(complete=True)


@register
def theme_minimal() -> Theme:
    """Complete theme without axis lines or ticks; major grid only."""
    return Theme(axis_line=False, ticks=False, panel_grid=True, grid_opacity=0.2, complete=True)


@register
def theme_classic() -> Theme:
    """Complete theme with axis lines and ticks and no grid."""
    return Theme(axis_line=True, ticks=True, panel_grid=False, complete=True)


def theme(**elements) -> Theme:
    """Partial theme: override individual :class:`Theme` fields, e.g.
    ``theme(text_engine="tex", legend_position="bottom")``.

    Added to a plot it is merged onto the current theme (or the default theme).

    Not registered as a chaining component: ``plot.theme`` is the plot's theme
    field, so use ``plot.add(theme(...))``.
    """
    unknown = set(elements) - _FIELD_NAMES
    if unknown:
        raise TypeError(
            f"unknown theme element(s) {sorted(unknown)}; valid: {sorted(_FIELD_NAMES)}"
        )
    return Theme(**elements, complete=False, overrides=frozenset(elements))
