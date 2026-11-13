"""Themes: the non-data appearance of a plot.

A :class:`Theme` is an immutable (frozen) dataclass. Complete themes set
``complete=True`` and *replace* the plot's theme when added:

* :func:`theme_manim` (= :func:`theme_default`): tuned for manim's black scene
  background; foreground derived from ``config.background_color``.
* ggplot2 looks on a white plot background: :func:`theme_gray` (alias
  :func:`theme_grey`), :func:`theme_bw`, :func:`theme_linedraw`,
  :func:`theme_light`, :func:`theme_dark`.
* Background-agnostic (drawn on the scene background, foreground from config):
  :func:`theme_minimal`, :func:`theme_classic`, :func:`theme_void`.

To drop the white "paper" of a ggplot look, add ``theme(plot_background=None,
foreground=None, axis_text_color=None, axis_title_color=None, title_color=None)``.

:func:`theme` builds a *partial* theme whose explicitly given fields are merged onto the
current theme (``old.merge(new)``, see :meth:`ggmanim.plot.ggplot._with_theme`).
A partial theme added to a plot without a theme is the default theme with the
overrides applied.

Nothing here reads manim's global ``config`` at import time: the foreground
colour (when ``foreground`` is ``None``) and the default panel size are resolved
when a plot is built, via :meth:`Theme.get_foreground` and :meth:`Theme.panel_size`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import manim
import numpy as np
from manim import BLACK, WHITE, ManimColor, Mobject, config

from .registry import register

__all__ = [
    "Theme",
    "theme",
    "theme_default",
    "theme_minimal",
    "theme_classic",
    "theme_manim",
    "theme_gray",
    "theme_grey",
    "theme_bw",
    "theme_linedraw",
    "theme_light",
    "theme_dark",
    "theme_void",
    "foreground_for_background",
]

_TEXT_ENGINES = ("text", "tex")
_LEGEND_POSITIONS = ("right", "left", "top", "bottom", "none")
_TITLE_POSITIONS = ("panel", "plot")
# per-axis element -> the global switch it inherits when its own switch is None
_AXIS_PARENTS = {
    "axis_text": "tick_labels",
    "axis_title": "axis_title",
    "axis_ticks": "ticks",
    "axis_line": "axis_line",
}
# text element -> (size field, colour field, font field)
_TEXT_ELEMENTS = {
    "axis_text": ("axis_text_size", "axis_text_color", "axis_text_font"),
    "axis_title": ("axis_title_size", "axis_title_color", "axis_title_font"),
    "title": ("title_size", "title_color", "title_font"),
    "subtitle": ("axis_title_size", "title_color", "title_font"),
    "caption": ("axis_text_size", "axis_text_color", "title_font"),
}
_COLOR_FIELDS = (
    "foreground",
    "axis_line_color",
    "axis_ticks_color",
    "axis_text_color",
    "axis_title_color",
    "title_color",
    "grid_color",
    "panel_background",
    "panel_border_color",
    "plot_background",
)
_UNIT_FIELDS = (
    "grid_opacity",
    "grid_minor_opacity",
    "panel_background_opacity",
    "plot_background_opacity",
    "axis_text_x_hjust",
    "axis_text_x_vjust",
    "axis_text_y_hjust",
    "axis_text_y_vjust",
)
# ggplot grey levels used by the complete themes
_GREY20, _GREY30, _GREY42, _GREY50 = "#333333", "#4D4D4D", "#6B6B6B", "#7F7F7F"
_GREY70, _GREY87, _GREY92 = "#B3B3B3", "#DEDEDE", "#EBEBEB"


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
        Per-element font families ``axis_text_font``, ``axis_title_font`` and
        ``title_font`` (title, subtitle, caption); ``None`` = ``font``. Fonts are
        ignored by the ``"tex"`` engine.
    Colour
        ``foreground``: colour for axes, ticks, legends and text; ``None`` derives it
        from ``config.background_color`` at build time (BLACK on light, WHITE on dark).
        Overrides (``None`` = ``foreground``): ``axis_line_color``,
        ``axis_ticks_color`` (``None`` = ``axis_line_color``), ``axis_text_color``
        (tick labels and caption), ``axis_title_color``, ``title_color`` (title and
        subtitle), ``grid_color``, ``panel_border_color``.
    Axes
        ``axis_line``: draw the axis lines; ``axis_stroke_width``; ``ticks``: draw tick
        marks; ``tick_length`` (scene units, drawn outward); ``tick_labels``: draw tick
        labels; ``axis_title``: draw the axis titles; ``tick_label_buff`` (gap between
        tick and label); ``axis_title_buff``.

        Per-axis switches ``axis_text_x``/``axis_text_y``, ``axis_title_x``/
        ``axis_title_y``, ``axis_ticks_x``/``axis_ticks_y`` and ``axis_line_x``/
        ``axis_line_y``: ``None`` inherits the global switch (``tick_labels``,
        ``axis_title``, ``ticks``, ``axis_line``); ``False`` blanks that element on
        one axis only, ``True`` forces it on (see :meth:`axis_element`).

        ``axis_text_x_angle``/``axis_text_y_angle``: tick-label rotation in degrees
        (counter-clockwise). ``axis_text_x_hjust``/``axis_text_x_vjust`` and
        ``axis_text_y_hjust``/``axis_text_y_vjust``, in ``[0, 1]``, pick the point of the *unrotated* label that is
        pinned next to its tick (``hjust`` 0 = left, 1 = right; ``vjust`` 0 = bottom,
        1 = top), as in ggplot2. ``None`` chooses them from the angle so the label
        hangs off the axis without crossing it (e.g. ``axis_text_x_angle=45``
        right-aligns each label's end at its tick); see :meth:`axis_text_just`.
    Panel
        ``panel_background``: fill colour behind the grid (``None`` = none),
        ``panel_background_opacity``. ``panel_border``: draw a rectangle round the
        panel in ``panel_border_color`` with ``panel_border_width``.
    Grid
        ``panel_grid_major``: draw major grid lines at the breaks;
        ``panel_grid_major_x``/``panel_grid_major_y`` switch one direction
        (``None`` = inherit; ``_x`` are the vertical lines at the x breaks).
        ``grid_stroke_width``, ``grid_opacity``. ``panel_grid_minor``: draw minor grid
        lines between the breaks (when the coord supports minor breaks) with
        ``grid_minor_stroke_width`` and ``grid_minor_opacity``. Both use ``grid_color``.
        ``panel_grid`` is a backward-compatible alias: ``theme(panel_grid=b)`` sets
        major and minor; reading ``Theme.panel_grid`` gives ``panel_grid_major``.
    Plot
        ``plot_background``: fill colour of a rectangle behind the whole plot (panel,
        titles, legend) padded by ``plot_margin`` (``None`` = none);
        ``plot_background_opacity``. ``plot_margin``: space round the plot in scene
        units, a number or ``(top, right, bottom, left)`` (stored as the 4-tuple); the
        built plot is fitted into the frame inside these margins.
        ``plot_title_position``: ``"panel"`` (ggplot's default) aligns the title and
        subtitle (left) and caption (right) with the panel area; ``"plot"`` with the
        whole plot including axis labels and legend.
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
    axis_text_font: str | None = None
    axis_title_font: str | None = None
    title_font: str | None = None

    foreground: Any | None = None
    axis_line_color: Any | None = None
    axis_ticks_color: Any | None = None
    axis_text_color: Any | None = None
    axis_title_color: Any | None = None
    title_color: Any | None = None

    axis_line: bool = True
    axis_stroke_width: float = 2.0
    ticks: bool = True
    tick_length: float = 0.12
    tick_labels: bool = True
    tick_label_buff: float = 0.1
    axis_title_buff: float = 0.25
    axis_title: bool = True

    axis_text_x: bool | None = None
    axis_text_y: bool | None = None
    axis_title_x: bool | None = None
    axis_title_y: bool | None = None
    axis_ticks_x: bool | None = None
    axis_ticks_y: bool | None = None
    axis_line_x: bool | None = None
    axis_line_y: bool | None = None

    axis_text_x_angle: float = 0.0
    axis_text_y_angle: float = 0.0
    axis_text_x_hjust: float | None = None
    axis_text_x_vjust: float | None = None
    axis_text_y_hjust: float | None = None
    axis_text_y_vjust: float | None = None

    panel_background: Any | None = None
    panel_background_opacity: float = 1.0
    panel_border: bool = False
    panel_border_color: Any | None = None
    panel_border_width: float = 2.0

    panel_grid_major: bool = True
    panel_grid_major_x: bool | None = None
    panel_grid_major_y: bool | None = None
    panel_grid_minor: bool = True
    grid_color: Any | None = None
    grid_stroke_width: float = 1.0
    grid_opacity: float = 0.15
    grid_minor_stroke_width: float = 0.5
    grid_minor_opacity: float = 0.08

    plot_background: Any | None = None
    plot_background_opacity: float = 1.0
    plot_margin: Any = 0.2
    plot_title_position: str = "panel"

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
        if self.plot_title_position not in _TITLE_POSITIONS:
            raise ValueError(
                f"plot_title_position must be one of {_TITLE_POSITIONS}, "
                f"got {self.plot_title_position!r}"
            )
        for name in _COLOR_FIELDS:
            value = getattr(self, name)
            if value is not None:
                try:
                    ManimColor(value)
                except Exception as exc:
                    raise ValueError(f"{name}: not a colour: {value!r}") from exc
        for name in _UNIT_FIELDS:
            value = getattr(self, name)
            if value is not None and not (_is_number(value) and 0 <= value <= 1):
                raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
        for name in ("axis_text_x_angle", "axis_text_y_angle"):
            angle = getattr(self, name)
            if not (_is_number(angle) and np.isfinite(angle)):
                raise ValueError(f"{name} must be a finite number of degrees, got {angle!r}")
        object.__setattr__(self, "plot_margin", _margin4(self.plot_margin))

    # -- backward compatibility ----------------------------------------------
    @property
    def panel_grid(self) -> bool:
        """Alias of ``panel_grid_major`` (read-only; ``theme(panel_grid=...)`` sets it)."""
        return self.panel_grid_major

    # -- element resolution --------------------------------------------------
    def axis_element(self, element: str, axis: str) -> bool:
        """Whether ``element`` (``"axis_text"``, ``"axis_title"``, ``"axis_ticks"``
        or ``"axis_line"``) is drawn on ``axis`` (``"x"``/``"y"``): the per-axis
        switch, or the global one when that is ``None``."""
        if element not in _AXIS_PARENTS or axis not in ("x", "y"):
            raise ValueError(f"unknown axis element {element!r} / axis {axis!r}")
        own = getattr(self, f"{element}_{axis}")
        return bool(getattr(self, _AXIS_PARENTS[element]) if own is None else own)

    def grid_major(self, axis: str) -> bool:
        """Whether major grid lines at the ``axis`` (``"x"``/``"y"``) breaks are drawn."""
        own = getattr(self, f"panel_grid_major_{axis}")
        return bool(self.panel_grid_major if own is None else own)

    def color_of(self, name: str) -> ManimColor:
        """A colour field resolved through its fallbacks (``axis_ticks_color`` ->
        ``axis_line_color``), finally :meth:`get_foreground`."""
        value = getattr(self, name)
        if value is None and name == "axis_ticks_color":
            value = self.axis_line_color
        return ManimColor(value) if value is not None else self.get_foreground()

    def text_style(self, element: str) -> dict[str, Any]:
        """``{"font_size", "color"[, "font"]}`` for a text element: ``"axis_text"``,
        ``"axis_title"``, ``"title"``, ``"subtitle"`` or ``"caption"``."""
        size, color, font = _TEXT_ELEMENTS[element]
        style: dict[str, Any] = {"font_size": getattr(self, size), "color": self.color_of(color)}
        family = getattr(self, font)
        if family is not None:
            style["font"] = family
        return style

    def axis_text_just(self, axis: str) -> tuple[float, float, bool]:
        """``(hjust, vjust, auto)`` for the ``axis`` (``"x"``/``"y"``) tick labels.

        Unset justifications follow the angle: x labels pin their top centre
        (angle 0), right end (angle > 0) or left end (angle < 0) to the tick; y
        labels their right middle (``|angle| < 45``) or the edge facing the axis
        (steeper). ``auto`` is True when neither was given; the coord then also
        nudges the rotated label so no part of it crosses back over the tick.
        """
        angle = _norm_angle(getattr(self, f"axis_text_{axis}_angle"))
        hj = getattr(self, f"axis_text_{axis}_hjust")
        vj = getattr(self, f"axis_text_{axis}_vjust")
        if axis == "x":
            if angle == 0:
                auto_h, auto_v = 0.5, 1.0
            else:
                auto_h, auto_v = (1.0 if angle > 0 else 0.0), 0.5
        elif abs(angle) >= 45:
            auto_h, auto_v = 0.5, (0.0 if angle > 0 else 1.0)
        else:
            auto_h, auto_v = 1.0, 0.5
        return (
            float(auto_h if hj is None else hj),
            float(auto_v if vj is None else vj),
            hj is None and vj is None,
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
        """A copy with ``changes`` applied (validated; ``panel_grid`` is expanded)."""
        return dataclasses.replace(self, **_alias(changes))

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
        to :meth:`get_foreground`. A ``font`` keyword is dropped for ``"tex"``.
        """
        if isinstance(content, Mobject):
            return content
        text = str(content)
        if font_size is None:
            font_size = self.axis_text_size
        if color is None:
            color = self.get_foreground()
        if self.text_engine == "tex":
            kwargs.pop("font", None)  # Pango font families do not apply to LaTeX
            return manim.Tex(text, font_size=font_size, color=color, **kwargs)
        if self.font is not None:
            kwargs.setdefault("font", self.font)
        return manim.Text(text, font_size=font_size, color=color, **kwargs)


_FIELD_NAMES = frozenset(f.name for f in dataclasses.fields(Theme)) - {"complete", "overrides"}
_ALIASES = {"panel_grid": ("panel_grid_major", "panel_grid_minor")}


def _alias(elements: dict[str, Any]) -> dict[str, Any]:
    """Expand alias fields (``panel_grid`` -> major and minor); explicit fields win."""
    out = dict(elements)
    for alias, targets in _ALIASES.items():
        if alias in out:
            value = out.pop(alias)
            for target in targets:
                out.setdefault(target, value)
    return out


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _norm_angle(angle: float) -> float:
    """Degrees normalised to ``(-180, 180]``."""
    a = float(angle) % 360.0
    return a - 360.0 if a > 180.0 else a


def _margin4(margin: Any) -> tuple[float, float, float, float]:
    """``plot_margin`` as ``(top, right, bottom, left)`` non-negative floats."""
    if _is_number(margin):
        values = [float(margin)] * 4
    elif isinstance(margin, Sequence) and not isinstance(margin, str) and len(margin) == 4:
        if not all(_is_number(m) for m in margin):
            raise ValueError(f"plot_margin entries must be numbers, got {margin!r}")
        values = [float(m) for m in margin]
    else:
        raise ValueError(
            f"plot_margin must be a number or (top, right, bottom, left), got {margin!r}"
        )
    if not all(np.isfinite(v) and v >= 0 for v in values):
        raise ValueError(f"plot_margin must be finite and >= 0, got {margin!r}")
    return values[0], values[1], values[2], values[3]


@register
def theme_default() -> Theme:
    """Default complete theme (same as :func:`theme_manim`): axis lines, outward
    ticks, faint major grid, colours following the scene background."""
    return Theme(complete=True)


@register
def theme_manim() -> Theme:
    """Complete theme tuned for manim's black background: no plot background,
    foreground (axes, ticks, text) derived from ``config.background_color``, axis
    lines with outward ticks and a faint grid. The default appearance."""
    return Theme(complete=True)


def _ggplot_look(**changes: Any) -> Theme:
    """A ggplot2-style complete theme on a white plot background."""
    base: dict[str, Any] = dict(
        plot_background=WHITE,
        foreground=_GREY20,
        axis_text_color=_GREY30,
        axis_title_color=BLACK,
        title_color=BLACK,
        axis_line=False,
        grid_opacity=1.0,
        grid_minor_opacity=1.0,
        complete=True,
    )
    base.update(changes)
    return Theme(**base)


@register
def theme_gray() -> Theme:
    """ggplot2's default look: grey panel, white grid lines, no axis lines, on a
    white plot background."""
    return _ggplot_look(panel_background=_GREY92, grid_color=WHITE)


@register
def theme_grey() -> Theme:
    """Alias of :func:`theme_gray`."""
    return theme_gray()


@register
def theme_bw() -> Theme:
    """White panel with a dark border and light grey grid lines."""
    return _ggplot_look(
        panel_background=WHITE,
        panel_border=True,
        panel_border_color=_GREY20,
        grid_color=_GREY92,
    )


@register
def theme_linedraw() -> Theme:
    """White panel, black border, thin black grid lines, black ticks and text."""
    return _ggplot_look(
        foreground=BLACK,
        axis_text_color=BLACK,
        panel_background=WHITE,
        panel_border=True,
        panel_border_color=BLACK,
        grid_color=BLACK,
        grid_stroke_width=0.5,
        grid_minor_stroke_width=0.25,
    )


@register
def theme_light() -> Theme:
    """White panel with a light grey border, light grey ticks and thin grid lines."""
    return _ggplot_look(
        panel_background=WHITE,
        panel_border=True,
        panel_border_color=_GREY70,
        axis_ticks_color=_GREY70,
        grid_color=_GREY87,
        grid_stroke_width=0.75,
        grid_minor_stroke_width=0.4,
    )


@register
def theme_dark() -> Theme:
    """Dark grey panel with darker grid lines, for bright data colours."""
    return _ggplot_look(
        panel_background=_GREY50,
        grid_color=_GREY42,
        grid_minor_stroke_width=0.4,
    )


@register
def theme_minimal() -> Theme:
    """Complete theme without axis lines or ticks; major grid only."""
    return Theme(
        axis_line=False, ticks=False, panel_grid_major=True, grid_opacity=0.2, complete=True
    )


@register
def theme_classic() -> Theme:
    """Complete theme with axis lines and ticks and no grid."""
    return Theme(
        axis_line=True, ticks=True, panel_grid_major=False, panel_grid_minor=False, complete=True
    )


@register
def theme_void() -> Theme:
    """Complete theme with only the data: no axis lines, ticks, tick labels, axis
    titles, grid or backgrounds (plot titles and legends are still drawn)."""
    return Theme(
        axis_line=False,
        ticks=False,
        tick_labels=False,
        axis_title=False,
        panel_grid_major=False,
        panel_grid_minor=False,
        complete=True,
    )


def theme(**elements) -> Theme:
    """Partial theme: override individual :class:`Theme` fields, e.g.
    ``theme(text_engine="tex", legend_position="bottom")``.

    Added to a plot it is merged onto the current theme (or the default theme).
    ``panel_grid=b`` is accepted as shorthand for ``panel_grid_major=b,
    panel_grid_minor=b`` (explicit major/minor values win).

    Not registered as a chaining component: ``plot.theme`` is the plot's theme
    field, so use ``plot.add(theme(...))``.
    """
    unknown = set(elements) - _FIELD_NAMES - set(_ALIASES)
    if unknown:
        raise TypeError(
            f"unknown theme element(s) {sorted(unknown)}; valid: {sorted(_FIELD_NAMES)}"
        )
    elements = _alias(elements)
    return Theme(**elements, complete=False, overrides=frozenset(elements))
