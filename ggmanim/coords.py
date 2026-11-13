"""Coordinate systems: turn trained position scales into a manim panel.

The entry point used by the build pipeline is :meth:`CoordCartesian.build`::

    coord = plot.coord or CoordCartesian()
    panel = coord.build(x_scale, y_scale, theme, labels, has_legend=...)

    panel.axes            # one manim Axes over the *transformed-space* expanded limits
    panel.transform(x, y) # (N, 3) scene points; x, y in transformed (scale) space
    panel.panel_params    # PanelParams(x_range, y_range, x_breaks, y_breaks, x_labels, y_labels)
    panel.x_ticks, panel.y_ticks              # VGroup of tick marks (one per break)
    panel.x_tick_labels, panel.y_tick_labels  # VGroup of labels (one per break, same order)
    panel.grid                                # VGroup of major grid lines
    panel.x_title, panel.y_title              # Mobject or None

Geoms (``ggm-s11.6``) receive data already mapped by the scales (i.e. in
transformed space: ``log10(x)`` for ``scale_x_log10``) and call
``panel.transform(xs, ys)`` once per group. Tick labels are the scale's
data-space label strings (``"10"``, ``"100"`` ...) placed at the transformed
breaks on a *linear* manim axis: we never use manim's ``LogBase`` or its numeric
labels.

``panel.transform`` reads the live axes, so it stays correct after the panel (or
the plot containing it) is moved or scaled.

Building is pure: scales, theme and caller-supplied label mobjects are not
mutated (title mobjects are copied before being positioned/rotated), and two
builds from the same inputs give identically positioned mobjects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from manim import DOWN, LEFT, PI, Axes, Line, Mobject, VGroup, VMobject

from .registry import register
from .scales.breaks import expand_range
from .theme import Theme, theme_default

__all__ = ["CoordCartesian", "coord_cartesian", "PanelParams", "CartesianPanel", "PanelAxes"]

_FALLBACK_RANGE = (0.0, 1.0)


@dataclass(frozen=True)
class PanelParams:
    """Resolved panel geometry, all positions in transformed (scale) space.

    ``x_range``/``y_range`` are the drawn axis ranges (limits incl. expansion);
    ``x_breaks``/``y_breaks`` the major tick positions within them and
    ``x_labels``/``y_labels`` the matching data-space label strings.
    """

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    x_breaks: tuple[float, ...]
    y_breaks: tuple[float, ...]
    x_labels: tuple[str, ...]
    y_labels: tuple[str, ...]


class PanelAxes(Axes):
    """manim ``Axes`` whose lines always meet at the bottom-left corner
    ``(x_min, y_min)`` (ggplot style) instead of at 0 when 0 is in range.

    Overriding ``_origin_shift`` keeps ``c2p`` consistent with where the lines are.
    """

    @staticmethod
    def _origin_shift(axis_range: Sequence[float]) -> float:
        return axis_range[0]


class CartesianPanel(VGroup):
    """The built panel: a VGroup of grid, axes, ticks, tick labels and axis titles.

    See the module docstring for the attributes. Submobject order (back to front):
    ``grid, axes, x_ticks, y_ticks, x_tick_labels, y_tick_labels[, x_title][, y_title]``.
    """

    def __init__(
        self,
        axes: Axes,
        panel_params: PanelParams,
        grid: VGroup,
        x_ticks: VGroup,
        y_ticks: VGroup,
        x_tick_labels: VGroup,
        y_tick_labels: VGroup,
        x_title: Mobject | None,
        y_title: Mobject | None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.axes = axes
        self.panel_params = panel_params
        self.grid = grid
        self.x_ticks = x_ticks
        self.y_ticks = y_ticks
        self.x_tick_labels = x_tick_labels
        self.y_tick_labels = y_tick_labels
        self.x_title = x_title
        self.y_title = y_title
        self.add(grid, axes, x_ticks, y_ticks, x_tick_labels, y_tick_labels)
        if x_title is not None:
            self.add(x_title)  # type: ignore[arg-type]
        if y_title is not None:
            self.add(y_title)  # type: ignore[arg-type]

    def transform(self, x: Any, y: Any) -> np.ndarray:
        """Transformed-space ``(x, y)`` -> ``(N, 3)`` scene points (one vectorised
        ``axes.c2p`` call). ``x`` and ``y`` broadcast; NaN in gives NaN out."""
        return _c2p(self.axes, x, y)


def _c2p(axes: Axes, x: Any, y: Any) -> np.ndarray:
    xs, ys = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    xs, ys = xs.ravel(), ys.ravel()
    if xs.size == 0:
        return np.zeros((0, 3))
    pts = np.asarray(axes.c2p(xs, ys), dtype=float)  # (3, N) for array input
    return pts.reshape(3, -1).T.copy()


def _range(scale, lim, expand: bool = True) -> tuple[float, float]:
    """Expanded transformed-space range for ``scale``; ``lim`` is an optional
    data-space zoom ``(lo, hi)`` (either end ``None`` = the scale's own limit)."""
    rng = None
    if scale is not None and lim is None:
        rng = scale.get_expanded_limits() if expand else scale.get_expanded_limits(False)
    elif scale is not None:
        known = scale.get_limits() or _FALLBACK_RANGE
        inc = scale._is_increasing() if hasattr(scale, "_is_increasing") else True
        t = np.asarray(scale.transform([np.nan if v is None else v for v in lim]), dtype=float)
        # lim[0] is the smaller data value: the low end for increasing transforms
        ends = [
            t[0] if np.isfinite(t[0]) else (known[0] if inc else known[1]),
            t[1] if np.isfinite(t[1]) else (known[1] if inc else known[0]),
        ]
        rng = expand_range(tuple(sorted(ends)), getattr(scale, "expand", None) if expand else False)
    if rng is None or not all(np.isfinite(rng)):
        rng = _FALLBACK_RANGE
    lo, hi = float(min(rng)), float(max(rng))
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    return lo, hi


def _breaks_labels(scale, rng) -> tuple[tuple[float, ...], tuple[str, ...]]:
    if scale is None:
        return (), ()
    breaks = np.asarray(scale.get_breaks(rng), dtype=float).ravel()
    labels = list(scale.get_labels(rng))
    if len(labels) != len(breaks):
        raise ValueError(f"scale {scale!r} returned {len(labels)} labels for {len(breaks)} breaks")
    return tuple(float(b) for b in breaks), tuple(str(lab) for lab in labels)


def _title(scale, labels: Mapping[str, Any] | None, aes: str):
    """Scale name wins (ggplot2), then ``labels[aes]``; ``None`` -> no title."""
    name = getattr(scale, "name", None) if scale is not None else None
    if name is not None:
        return name
    if labels is None:
        return None
    return labels.get(aes)


class CoordCartesian:
    """Cartesian coordinates.

    ``xlim``/``ylim`` (data space, optional) *zoom* the panel without touching the
    data: no values are censored or dropped (unlike ``xlim()``/``scale limits``),
    geoms simply draw outside the visible range (Phase 1 does no clipping).
    ``expand=False`` turns off the scales' expansion (for zoomed and unzoomed axes).
    """

    def __init__(
        self,
        xlim: Sequence[float | None] | None = None,
        ylim: Sequence[float | None] | None = None,
        expand: bool = True,
    ):
        for name, lim in (("xlim", xlim), ("ylim", ylim)):
            if lim is not None and len(tuple(lim)) != 2:
                raise ValueError(f"{name} must be (lo, hi), got {lim!r}")
        self.xlim = None if xlim is None else tuple(xlim)
        self.ylim = None if ylim is None else tuple(ylim)
        self.expand = bool(expand)

    def add_to(self, plot):
        return plot._with_coord(self)

    def __repr__(self):
        return f"coord_cartesian(xlim={self.xlim!r}, ylim={self.ylim!r}, expand={self.expand!r})"

    # -- geometry --------------------------------------------------------------
    def _axis_range(self, scale, lim) -> tuple[float, float]:
        return _range(scale, lim, self.expand)

    def setup_panel_params(self, x_scale, y_scale) -> PanelParams:
        """Ranges, breaks and labels for trained ``x_scale``/``y_scale`` (either may
        be ``None``: range ``(0, 1)``, no breaks). Does not mutate the scales."""
        xr = self._axis_range(x_scale, self.xlim)
        yr = self._axis_range(y_scale, self.ylim)
        xb, xl = _breaks_labels(x_scale, xr)
        yb, yl = _breaks_labels(y_scale, yr)
        return PanelParams(
            x_range=xr, y_range=yr, x_breaks=xb, y_breaks=yb, x_labels=xl, y_labels=yl
        )

    def build(
        self,
        x_scale,
        y_scale,
        theme: Theme | None = None,
        labels: Mapping[str, Any] | None = None,
        *,
        has_legend: bool = False,
    ) -> CartesianPanel:
        """Build the panel mobject.

        Parameters
        ----------
        x_scale, y_scale : trained continuous position scales (or ``None``).
        theme : :class:`Theme` (default :func:`theme_default`).
        labels : mapping with optional ``"x"``/``"y"`` titles (str or Mobject; ``None``
            suppresses). A scale's ``name`` takes precedence. Mobjects are copied.
        has_legend : reserve room for a legend in the default panel size.
        """
        theme = theme if theme is not None else theme_default()
        pp = self.setup_panel_params(x_scale, y_scale)
        fg = theme.get_foreground()
        width, height = theme.panel_size(has_legend)
        (x0, x1), (y0, y1) = pp.x_range, pp.y_range

        axes = PanelAxes(
            x_range=[x0, x1, x1 - x0],
            y_range=[y0, y1, y1 - y0],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={
                "include_ticks": False,
                "include_numbers": False,
                "stroke_width": theme.axis_stroke_width,
            },
        )
        axes.set_stroke(
            color=fg, width=theme.axis_stroke_width, opacity=1.0 if theme.axis_line else 0.0
        )

        xb = np.asarray(pp.x_breaks, dtype=float)
        yb = np.asarray(pp.y_breaks, dtype=float)
        bottom = _c2p(axes, xb, y0)
        top = _c2p(axes, xb, y1)
        left = _c2p(axes, x0, yb)
        right = _c2p(axes, x1, yb)

        grid = VGroup()
        if theme.panel_grid:
            for a, b in list(zip(bottom, top)) + list(zip(left, right)):
                grid.add(
                    Line(
                        a,
                        b,
                        stroke_width=theme.grid_stroke_width,
                        stroke_color=fg,
                        stroke_opacity=theme.grid_opacity,
                    )
                )

        tl = theme.tick_length
        x_ticks, y_ticks = VGroup(), VGroup()
        if theme.ticks:
            for p in bottom:
                x_ticks.add(
                    Line(p, p + tl * DOWN, stroke_width=theme.axis_stroke_width, stroke_color=fg)
                )
            for p in left:
                y_ticks.add(
                    Line(p, p + tl * LEFT, stroke_width=theme.axis_stroke_width, stroke_color=fg)
                )

        gap = (tl if theme.ticks else 0.0) + theme.tick_label_buff
        x_tick_labels, y_tick_labels = VGroup(), VGroup()
        if theme.tick_labels:
            for p, s in zip(bottom, pp.x_labels):
                x_tick_labels.add(_label(theme, s).next_to(p, DOWN, buff=gap))
            for p, s in zip(left, pp.y_labels):
                y_tick_labels.add(_label(theme, s).next_to(p, LEFT, buff=gap))

        x_title = _make_title(theme, _title(x_scale, labels, "x"))
        y_title = _make_title(theme, _title(y_scale, labels, "y"))
        corner = _c2p(axes, x0, y0)[0]
        centre = _c2p(axes, (x0 + x1) / 2, (y0 + y1) / 2)[0]
        if x_title is not None:
            below = (
                x_tick_labels if len(x_tick_labels) else VGroup(Line(corner, corner + gap * DOWN))
            )
            x_title.next_to(below, DOWN, buff=theme.axis_title_buff)
            x_title.set_x(centre[0])
        if y_title is not None:
            y_title.rotate(PI / 2)
            beside = (
                y_tick_labels if len(y_tick_labels) else VGroup(Line(corner, corner + gap * LEFT))
            )
            y_title.next_to(beside, LEFT, buff=theme.axis_title_buff)
            y_title.set_y(centre[1])

        return CartesianPanel(
            axes, pp, grid, x_ticks, y_ticks, x_tick_labels, y_tick_labels, x_title, y_title
        )


def _label(theme: Theme, s: str) -> Mobject:
    if s == "":
        return VMobject()  # keep one label per break; nothing to draw
    return theme.make_text(s, font_size=theme.axis_text_size)


def _make_title(theme: Theme, content: Any) -> Mobject | None:
    if content is None:
        return None
    if isinstance(content, Mobject):
        return content.copy()  # never move/rotate the caller's mobject
    if str(content) == "":
        return None
    return theme.make_text(content, font_size=theme.axis_title_size)


@register
def coord_cartesian(xlim=None, ylim=None, expand: bool = True) -> CoordCartesian:
    """Cartesian coordinates; ``xlim``/``ylim`` zoom without censoring data."""
    return CoordCartesian(xlim=xlim, ylim=ylim, expand=expand)
