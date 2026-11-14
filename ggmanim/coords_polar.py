"""``coord_polar``: polar coordinates (ggplot2's ``coord_polar``).

One position aesthetic becomes the angle (``theta``), the other the radius::

    # pie: one stacked bar of width 1, its y (count) mapped to the angle
    ggplot(df, aes(x=1, fill="kind")).geom_bar(width=1).add(coord_polar(theta="y"))
    # coxcomb / rose: bars around the circle, heights as radii
    ggplot(df, aes("month", "n")).geom_col(width=1).add(coord_polar())

* The theta range (the theta scale's limits; no expansion for a continuous
  scale, half a level each side for a discrete one, as in ggplot2) maps onto one
  full turn, starting at 12 o'clock + ``start`` radians, clockwise for
  ``direction=1``. The radius range (unexpanded limits) maps onto
  ``[inner_radius, 1] * 0.4`` of the (square) panel's side: a donut comes from
  ``inner_radius > 0`` or from radius limits that start below the data
  (``xlim(0.5, 2.5)`` with bars at ``x=2``).
* :meth:`CoordPolar.build` returns a :class:`PolarPanel` (``is_linear`` false):
  geoms munch rects, paths and polygons (:func:`~ggmanim.coords.coord_munch`), so
  bars become wedges and lines become arcs/spirals.
  :meth:`PolarPanel.munch_distance` measures segments along the arc.
* :func:`coord_radar` is the same mapping with ``is_linear`` true: vertices are
  placed in polar position but joined by straight lines (radar / spider charts
  with ``geom_polygon(fill=None)``).

The panel is drawn like ggplot2's: concentric circles at the radius breaks
plus an outer one at 0.45, spokes at the theta breaks, theta labels around the
circle (``axis_text_x``), the radius axis (``axis_*_y``) on the left edge of
the panel (further left when theta labels reach past that edge), the theta title below and the radius title left. Grid lines
follow ``panel_grid_major_<aes>`` of the aesthetic they show.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from manim import DOWN, LEFT, PI, Circle, Line, Polygon, VGroup

from .coords import (
    _FALLBACK_RANGE,
    CartesianPanel,
    PanelAxes,
    PanelParams,
    _box,
    _breaks_labels,
    _c2p,
    _label,
    _make_title,
    _minor_breaks,
    _title,
)
from .registry import register
from .scales.breaks import expand_range
from .theme import Theme, theme_default

__all__ = ["PolarPanel", "CoordPolar", "coord_polar", "coord_radar"]

#: Outer radius of the data, as a fraction of the panel side (ggplot2: 0.4).
R_MAX = 0.4
#: Radius of the outer grid circle and of the theta labels (ggplot2: 0.45).
R_LABEL = 0.45
_TWO_PI = 2 * np.pi


class PolarPanel(CartesianPanel):
    """A :class:`CartesianPanel` for :func:`coord_polar`.

    ``axes`` is a square :class:`~ggmanim.coords.PanelAxes` over ``[0, 1]^2``
    (its lines hidden) that fixes the panel's position and size; the centre of
    the circle is ``(0.5, 0.5)``. ``transform(x, y)`` maps aesthetic-space
    ``(x, y)`` to the scene through the polar mapping; ``panel_params`` stay in
    aesthetic orientation (``x_range`` is the x aesthetic's range, whether it is
    the angle or the radius).

    ``x_tick_labels`` are the theta labels around the circle (``x_ticks`` is
    empty), ``y_ticks``/``y_tick_labels`` the radius axis; ``x_title`` is the
    theta title (below), ``y_title`` the radius title (left). ``theta`` is
    ``"x"`` or ``"y"``; ``theta_range``/``r_range`` the ranges mapped onto the
    turn and onto ``[inner, R_MAX]``.
    """

    is_linear = False
    theta: str = "x"
    start: float = 0.0
    direction: int = 1
    inner: float = 0.0
    theta_range: tuple[float, float] = (0.0, 1.0)
    r_range: tuple[float, float] = (0.0, 1.0)

    def _theta_r(self, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(angle fraction of the turn, radius in panel units)``, unrotated."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        t, r = (x, y) if self.theta == "x" else (y, x)
        t0, t1 = self.theta_range
        tn = (t - t0) / ((t1 - t0) or 1.0)
        return tn.ravel(), self.radius(r).ravel()

    def radius(self, r: Any) -> np.ndarray:
        """Radius-aesthetic values -> radii in panel units (``[inner, R_MAX]``)."""
        (r0, r1) = self.r_range
        r = np.asarray(r, dtype=float)
        return self.inner + (r - r0) / ((r1 - r0) or 1.0) * (R_MAX - self.inner)

    def angle(self, frac: Any) -> np.ndarray:
        """Screen angle (radians, clockwise from 12 o'clock) of a theta fraction."""
        return self.direction * (np.asarray(frac, dtype=float) * _TWO_PI + self.start)

    def polar_to_scene(self, frac: Any, rn: Any) -> np.ndarray:
        """``(theta fraction, radius in panel units)`` -> ``(N, 3)`` scene points."""
        a = self.angle(frac)
        rn = np.asarray(rn, dtype=float)
        return _c2p(self.axes, 0.5 + rn * np.sin(a), 0.5 + rn * np.cos(a))

    def transform(self, x: Any, y: Any) -> np.ndarray:
        """Aesthetic-space ``(x, y)`` -> ``(N, 3)`` scene points (NaN in, NaN out)."""
        tn, rn = self._theta_r(x, y)
        return self.polar_to_scene(tn, rn)

    def munch_distance(self, x: Any, y: Any) -> np.ndarray:
        """Length of each segment of ``(x, y)`` along its polar image, as a
        fraction of the panel side: ``hypot(dr, r_mean * dtheta)`` (the arc
        length for a constant radius, the radial distance for a constant angle)."""
        tn, rn = self._theta_r(x, y)
        dt = np.diff(tn) * _TWO_PI
        rm = (np.abs(rn[1:]) + np.abs(rn[:-1])) / 2
        return np.hypot(np.diff(rn), rm * dt)


def _theta_limits(scale: Any) -> tuple[float, float]:
    """Unexpanded continuous limits; discrete: half a level each side."""
    if scale is None:
        return _FALLBACK_RANGE
    discrete = hasattr(scale, "continuous_limits")
    expand = getattr(scale, "expand", None)
    if expand is None:
        expand = (0.0, 0.5) if discrete else False
    rng = scale.get_expanded_limits(expand)
    return _finite_range(rng)


def _r_limits(scale: Any) -> tuple[float, float]:
    """Unexpanded limits (ggplot2's default for the radius); ``scale.expand``
    if the scale sets one."""
    if scale is None:
        return _FALLBACK_RANGE
    expand = getattr(scale, "expand", None)
    rng = scale.get_expanded_limits(False if expand is None else expand)
    return _finite_range(rng)


def _finite_range(rng: Any) -> tuple[float, float]:
    if rng is None or not all(np.isfinite(rng)):
        rng = _FALLBACK_RANGE
    lo, hi = float(min(rng)), float(max(rng))
    if hi <= lo:
        lo, hi = expand_range((lo, hi), False)
    return lo, hi


def _in_range(
    breaks: tuple[float, ...], labels: tuple[str, ...], rng: tuple[float, float]
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    lo, hi = rng
    tol = 1e-9 * max(1.0, hi - lo)
    keep = [np.isfinite(b) and lo - tol <= b <= hi + tol for b in breaks]
    return (
        tuple(b for b, k in zip(breaks, keep) if k),
        tuple(s for s, k in zip(labels, keep) if k),
    )


class CoordPolar:
    """Polar coordinates (ggplot2's ``coord_polar``).

    Parameters
    ----------
    theta : ``"x"`` or ``"y"``
        The aesthetic mapped to the angle; the other is the radius. A pie chart
        is a single stacked bar with ``theta="y"``.
    start : float
        Offset of the starting angle from 12 o'clock, in radians.
    direction : ``1`` or ``-1``
        ``1`` clockwise, ``-1`` anticlockwise.
    clip : ``"on"`` or ``"off"``
        Accepted for compatibility; marks are never clipped (ggplot2's polar
        default is ``"off"``): data inside the scale limits lies inside the circle.
    inner_radius : float in ``[0, 1)``
        Radius of the hole as a fraction of the outer radius (donut charts; like
        ggplot2's ``coord_radial(inner.radius=)``).
    """

    is_linear = False

    def __init__(
        self,
        theta: str = "x",
        start: float = 0.0,
        direction: int = 1,
        clip: str = "off",
        inner_radius: float = 0.0,
    ):
        if theta not in ("x", "y"):
            raise ValueError(f"theta must be 'x' or 'y', got {theta!r}")
        if direction not in (1, -1):
            raise ValueError(f"direction must be 1 or -1, got {direction!r}")
        if clip not in ("on", "off"):
            raise ValueError(f"clip must be 'on' or 'off', got {clip!r}")
        inner_radius = float(inner_radius)
        if not 0.0 <= inner_radius < 1.0:
            raise ValueError(f"inner_radius must be in [0, 1), got {inner_radius!r}")
        self.theta = theta
        self.r = "y" if theta == "x" else "x"
        self.start = float(start)
        self.direction = int(direction)
        self.clip = clip
        self.inner_radius = inner_radius

    def add_to(self, plot):
        return plot._with_coord(self)

    def _name(self) -> str:
        return "coord_polar"

    def __repr__(self):
        extra = f", inner_radius={self.inner_radius!r}" if self.inner_radius else ""
        return (
            f"{self._name()}(theta={self.theta!r}, start={self.start!r}, "
            f"direction={self.direction!r}{extra})"
        )

    def setup_panel_params(self, x_scale, y_scale) -> PanelParams:
        """Aesthetic-orientation ranges (theta: the full turn; radius: unexpanded
        limits), breaks inside them and their labels."""
        scales = {"x": x_scale, "y": y_scale}
        rng = {
            self.theta: _theta_limits(scales[self.theta]),
            self.r: _r_limits(scales[self.r]),
        }
        bl = {a: _in_range(*_breaks_labels(scales[a], rng[a]), rng[a]) for a in ("x", "y")}
        return PanelParams(
            x_range=rng["x"],
            y_range=rng["y"],
            x_breaks=bl["x"][0],
            y_breaks=bl["y"][0],
            x_labels=bl["x"][1],
            y_labels=bl["y"][1],
            x_minor_breaks=_minor_breaks(x_scale, rng["x"]),
            y_minor_breaks=_minor_breaks(y_scale, rng["y"]),
        )

    def build(
        self,
        x_scale,
        y_scale,
        theme: Theme | None = None,
        labels: Mapping[str, Any] | None = None,
        *,
        has_legend: bool = False,
    ) -> PolarPanel:
        """Build the :class:`PolarPanel`: a square panel (side ``min`` of the
        theme's panel size) with circular grid, theta labels and radius axis."""
        theme = theme if theme is not None else theme_default()
        pp = self.setup_panel_params(x_scale, y_scale)
        side = min(theme.panel_size(has_legend))
        axes = PanelAxes(
            x_range=[0.0, 1.0, 1.0],
            y_range=[0.0, 1.0, 1.0],
            x_length=side,
            y_length=side,
            tips=False,
            axis_config={"include_ticks": False, "include_numbers": False},
        )
        axes.x_axis.set_stroke(opacity=0.0)
        axes.y_axis.set_stroke(opacity=0.0)

        th, ra = self.theta, self.r
        t_range, r_range = (pp.x_range, pp.y_range) if th == "x" else (pp.y_range, pp.x_range)
        t_breaks, r_breaks = (pp.x_breaks, pp.y_breaks) if th == "x" else (pp.y_breaks, pp.x_breaks)
        t_labels, r_labels = (pp.x_labels, pp.y_labels) if th == "x" else (pp.y_labels, pp.x_labels)
        t_minor = pp.x_minor_breaks if th == "x" else pp.y_minor_breaks

        # a geometry-only panel: transform() works from here on
        geo = PolarPanel(axes, pp, VGroup(), VGroup(), VGroup(), VGroup(), VGroup(), None, None)
        geo.theta, geo.start, geo.direction = th, self.start, self.direction
        geo.inner = self.inner_radius * R_MAX
        geo.theta_range, geo.r_range = t_range, r_range
        t0, t1 = t_range
        tfrac = (np.asarray(t_breaks, dtype=float) - t0) / ((t1 - t0) or 1.0)
        tminor = (np.asarray(t_minor, dtype=float) - t0) / ((t1 - t0) or 1.0)
        rn = geo.radius(r_breaks)
        centre = geo.polar_to_scene(0.0, 0.0)[0]

        # -- grid: spokes (theta) and circles (radius) ---------------------------
        color = theme.color_of("grid_color")
        major: dict[str, Any] = dict(
            stroke_width=theme.grid_stroke_width,
            stroke_color=color,
            stroke_opacity=theme.grid_opacity,
        )
        grid_major, grid_minor = VGroup(), VGroup()
        if theme.grid_major(th):
            for p in geo.polar_to_scene(tfrac, R_LABEL):
                grid_major.add(Line(centre, p, **major))
        if theme.grid_major(ra):
            for r in [*(v for v in rn if v > 1e-9), R_LABEL]:
                grid_major.add(
                    Circle(radius=r * side, arc_center=centre).set_stroke(
                        color=color, width=theme.grid_stroke_width, opacity=theme.grid_opacity
                    )
                )
        if getattr(theme, "panel_grid_minor", True) and len(tminor):
            keep = ~np.any(np.isclose(tminor[:, None] % 1.0, tfrac[None, :] % 1.0), axis=1)
            for p in geo.polar_to_scene(tminor[keep], R_LABEL):
                grid_minor.add(
                    Line(
                        centre,
                        p,
                        stroke_width=theme.grid_minor_stroke_width,
                        stroke_color=color,
                        stroke_opacity=theme.grid_minor_opacity,
                    )
                )
        grid = VGroup(*grid_minor, *grid_major)

        # -- theta labels around the circle ---------------------------------------
        t_lab = list(t_labels)
        tf = list(tfrac)
        if len(tf) > 1 and ((tf[-1] - tf[0]) % 1.0) * _TWO_PI < 0.05:
            t_lab[0] = f"{t_lab[0]}/{t_lab[-1]}" if t_lab[0] != t_lab[-1] else t_lab[0]
            t_lab, tf = t_lab[:-1], tf[:-1]
        x_tick_labels = VGroup()
        if theme.axis_element("axis_text", "x"):
            pts = geo.polar_to_scene(tf, R_LABEL)
            for p, s, a in zip(pts, t_lab, geo.angle(tf)):
                lab = _label(theme, s, "x")
                u = np.array([np.sin(a), np.cos(a), 0.0])
                reach = (abs(u[0]) * lab.width + abs(u[1]) * lab.height) / 2
                lab.move_to(p + u * (theme.tick_label_buff + reach))
                x_tick_labels.add(lab)  # type: ignore[arg-type]

        # -- radius axis on the left edge (left of theta labels that reach it) ---
        a, b = _c2p(axes, [0.0, 0.0], [0.5 + geo.inner, 0.5 + R_MAX])
        ax_x = a[0]
        for lab in x_tick_labels:
            if lab.get_bottom()[1] < b[1] and lab.get_top()[1] > a[1]:
                ax_x = min(ax_x, lab.get_left()[0] - theme.tick_label_buff)
        a[0] = b[0] = ax_x
        left = _c2p(axes, 0.0, 0.5 + np.asarray(rn, dtype=float))
        left[:, 0] = ax_x
        y_ticks, y_tick_labels = VGroup(), VGroup()
        line_color, tick_color = (
            theme.color_of("axis_line_color"),
            theme.color_of("axis_ticks_color"),
        )
        if theme.axis_element("axis_line", "y"):
            axes.add(Line(a, b, stroke_width=theme.axis_stroke_width, stroke_color=line_color))
        tl = theme.tick_length
        if theme.axis_element("axis_ticks", "y"):
            for p in left:
                y_ticks.add(
                    Line(
                        p,
                        p + tl * LEFT,
                        stroke_width=theme.axis_stroke_width,
                        stroke_color=tick_color,
                    )
                )
        gap = (tl if len(y_ticks) else 0.0) + theme.tick_label_buff
        if theme.axis_element("axis_text", "y"):
            for p, s in zip(left, r_labels):
                y_tick_labels.add(_label(theme, s, "y").next_to(p, LEFT, buff=gap))  # type: ignore[arg-type]

        # -- titles: theta below, radius left ------------------------------------
        swapped = None if labels is None else {"x": labels.get(th), "y": labels.get(ra)}
        scales = {"x": x_scale, "y": y_scale}
        x_title = y_title = None
        if theme.axis_element("axis_title", "x"):
            x_title = _make_title(theme, _title(scales[th], swapped, "x"))
        if theme.axis_element("axis_title", "y"):
            y_title = _make_title(theme, _title(scales[ra], swapped, "y"))
        lo, hi = _c2p(axes, [0.0, 1.0], [0.0, 1.0])
        frame = VGroup(
            Line(lo, np.array([hi[0], lo[1], 0.0])), Line(lo, np.array([lo[0], hi[1], 0.0]))
        )
        if x_title is not None:
            x_title.next_to(VGroup(frame[0], *x_tick_labels), DOWN, buff=theme.axis_title_buff)
            x_title.set_x(centre[0])
        if y_title is not None:
            y_title.rotate(PI / 2)
            beside = VGroup(frame[1], *y_tick_labels, *x_tick_labels)
            y_title.next_to(beside, LEFT, buff=theme.axis_title_buff)
            y_title.set_y(centre[1])

        panel = PolarPanel(
            axes, pp, grid, VGroup(), y_ticks, x_tick_labels, y_tick_labels, x_title, y_title
        )
        panel.theta, panel.start, panel.direction = geo.theta, geo.start, geo.direction
        panel.inner, panel.theta_range, panel.r_range = geo.inner, geo.theta_range, geo.r_range
        panel.grid_major, panel.grid_minor = grid_major, grid_minor
        panel.is_linear = self.is_linear
        panel.clip = "off"
        _decorate(panel, theme, lo, hi)
        return panel


def _decorate(panel: PolarPanel, theme: Theme, lo: np.ndarray, hi: np.ndarray) -> None:
    """Panel background (the whole square, as in ggplot2) and border."""
    if theme.panel_background is not None:
        bg = Polygon(*_box(lo, hi), stroke_width=0)
        bg.set_fill(theme.panel_background, opacity=theme.panel_background_opacity)
        panel.background = bg
        panel.add_to_back(bg)
    if theme.panel_border:
        border = Polygon(
            *_box(lo, hi),
            stroke_color=theme.color_of("panel_border_color"),
            stroke_width=theme.panel_border_width,
        )
        border.set_fill(opacity=0)
        panel.border = border
        panel.add(border)


class _CoordRadar(CoordPolar):
    is_linear = True

    def _name(self) -> str:
        return "coord_radar"


@register
def coord_polar(
    theta: str = "x",
    start: float = 0.0,
    direction: int = 1,
    clip: str = "off",
    inner_radius: float = 0.0,
) -> CoordPolar:
    """Polar coordinates: ``theta`` (``"x"``/``"y"``) is the angle, the other
    aesthetic the radius. ``geom_bar(width=1) + coord_polar(theta="y")`` is a pie,
    ``geom_col(width=1) + coord_polar()`` a coxcomb; ``inner_radius`` (fraction of
    the outer radius) makes a donut. See :class:`CoordPolar`."""
    return CoordPolar(
        theta=theta, start=start, direction=direction, clip=clip, inner_radius=inner_radius
    )


@register
def coord_radar(
    theta: str = "x",
    start: float = 0.0,
    direction: int = 1,
    clip: str = "off",
    inner_radius: float = 0.0,
) -> CoordPolar:
    """:func:`coord_polar` with straight edges: vertices are placed in polar
    position but never munched, so ``geom_polygon(fill=None)`` draws a radar
    (spider) chart with straight spokes-to-spoke lines."""
    return _CoordRadar(
        theta=theta, start=start, direction=direction, clip=clip, inner_radius=inner_radius
    )
