"""Coordinate systems: turn trained position scales into a manim panel.

The entry point used by the build pipeline is :meth:`CoordCartesian.build`::

    coord = plot.coord or CoordCartesian()
    panel = coord.build(x_scale, y_scale, theme, labels, has_legend=...)

    panel.axes            # one manim Axes over the *transformed-space* expanded limits
    panel.transform(x, y) # (N, 3) scene points; x, y in transformed (scale) space
    panel.panel_params    # PanelParams(x_range, y_range, x_breaks, y_breaks, x_labels, y_labels)
    panel.x_ticks, panel.y_ticks              # VGroup of tick marks (one per break)
    panel.x_tick_labels, panel.y_tick_labels  # VGroup of labels (one per break, same order)
    panel.grid                                # VGroup of grid lines (minor, then major)
    panel.grid_major, panel.grid_minor        # the same split by kind (one GridLines,
                                              # a VMobject of all its lines, per axis)
    panel.x_title, panel.y_title              # Mobject or None

Geoms (``ggm-s11.6``) receive data already mapped by the scales (i.e. in
transformed space: ``log10(x)`` for ``scale_x_log10``) and call
``panel.transform(xs, ys)`` once per group. Tick labels are the scale's
data-space label strings (``"10"``, ``"100"`` ...) placed at the transformed
breaks on a *linear* manim axis: we never use manim's ``LogBase`` or its numeric
labels.

Minor grid lines (``scale.get_minor_breaks``) are drawn when
``theme.panel_grid_minor`` is true, styled by ``grid_minor_stroke_width`` /
``grid_minor_opacity``. A position scale's ``position`` (``"top"`` for x,
``"right"`` for y) moves its axis line, ticks, tick labels and title to that side.
Tick-label strings go through :func:`ggmanim.labels.render_label` (``"$...$"`` math
-> Unicode superscripts with the Text engine; LaTeX specials escaped with Tex).

``panel.transform`` reads the live axes, so it stays correct after the panel (or
the plot containing it) is moved or scaled.

Building is pure: scales, theme and caller-supplied label mobjects are not
mutated (title mobjects are copied before being positioned/rotated), and two
builds from the same inputs give identically positioned mobjects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from manim import (
    DEGREES,
    DL,
    DOWN,
    LEFT,
    PI,
    RIGHT,
    UP,
    UR,
    Axes,
    Line,
    ManimColor,
    Mobject,
    Polygon,
    VGroup,
    VMobject,
)

from .labels import render_label
from .registry import register
from .scales.breaks import expand_range
from .theme import Theme, hjust_align, theme_default

__all__ = [
    "CoordCartesian",
    "coord_cartesian",
    "PanelParams",
    "CartesianPanel",
    "PanelAxes",
    "CoordFlip",
    "coord_flip",
    "CoordFixed",
    "coord_fixed",
    "is_linear",
    "coord_munch",
    "fit_breaks",
]

_CLIP = ("on", "off")

_FALLBACK_RANGE = (0.0, 1.0)


# region: panel
@dataclass(frozen=True)
class PanelParams:
    """Resolved panel geometry, all positions in transformed (scale) space.

    ``x_range``/``y_range`` are the drawn axis ranges (limits incl. expansion);
    ``x_breaks``/``y_breaks`` the major tick positions within them and
    ``x_labels``/``y_labels`` the matching data-space label strings;
    ``x_minor_breaks``/``y_minor_breaks`` the minor grid positions.
    ``flipped`` is true under :func:`coord_flip` (the params stay in aesthetic
    orientation, so ``x_range`` is then the *vertical* axis' range).
    """

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    x_breaks: tuple[float, ...]
    y_breaks: tuple[float, ...]
    x_labels: tuple[str, ...]
    y_labels: tuple[str, ...]
    x_minor_breaks: tuple[float, ...] = ()
    y_minor_breaks: tuple[float, ...] = ()
    flipped: bool = False


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

    ``clip`` (``"on"``/``"off"``) is the coord's clipping policy, read by the geoms.
    ``flipped`` is true for :func:`coord_flip`: ``panel_params`` stay in aesthetic
    orientation (``x_range`` is the *x aesthetic's* range, drawn vertically) while
    ``axes``/``x_ticks``/``x_tick_labels``/``x_title`` are the horizontal ones.
    """

    clip: str = "on"
    flipped: bool = False

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
        self.grid_major: VGroup = grid  # build() splits these into major/minor
        self.grid_minor: VGroup = VGroup()
        self.x_ticks = x_ticks
        self.y_ticks = y_ticks
        self.x_tick_labels = x_tick_labels
        self.y_tick_labels = y_tick_labels
        self.x_title = x_title
        self.y_title = y_title
        self.background: VMobject | None = None  # theme.panel_background (set by build)
        self.border: VMobject | None = None  # theme.panel_border (set by build)
        self.add(grid, axes, x_ticks, y_ticks, x_tick_labels, y_tick_labels)
        if x_title is not None:
            self.add(x_title)  # type: ignore[arg-type]
        if y_title is not None:
            self.add(y_title)  # type: ignore[arg-type]

    def transform(self, x: Any, y: Any) -> np.ndarray:
        """Transformed-space ``(x, y)`` -> ``(N, 3)`` scene points (one vectorised
        ``axes.c2p`` call). ``x`` and ``y`` broadcast; NaN in gives NaN out.
        Under ``coord_flip`` the x aesthetic goes on the vertical axis."""
        if self.flipped:
            x, y = y, x
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
        # discrete position scales: get_limits() are levels; use numeric positions
        num = getattr(scale, "continuous_limits", None)
        known = (num() if num is not None else scale.get_limits()) or _FALLBACK_RANGE
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


def _minor_breaks(scale, rng) -> tuple[float, ...]:
    get = getattr(scale, "get_minor_breaks", None) if scale is not None else None
    if get is None:
        return ()
    mb = np.asarray(get(rng), dtype=float).ravel()
    lo, hi = rng
    return tuple(float(b) for b in mb if np.isfinite(b) and lo <= b <= hi)


def _side(scale, default: str) -> str:
    return getattr(scale, "position", None) or default


def _title(scale, labels: Mapping[str, Any] | None, aes: str):
    """Scale name wins (ggplot2), then ``labels[aes]``; ``None`` -> no title."""
    name = getattr(scale, "name", None) if scale is not None else None
    if name is not None:
        return name
    if labels is None:
        return None
    return labels.get(aes)


class GridLines(VMobject):
    """Straight lines from ``starts[i]`` to ``ends[i]`` (``(N, 3)`` scene points) as
    *one* VMobject with a subpath per line: the grid of one kind (major/minor) along
    one axis is a single mobject instead of one :class:`~manim.Line` per break,
    which is much cheaper to build, transform and render (ggm-bam.2).

    Each subpath has exactly the points ``Line(start, end)`` would have, so the
    lines draw the same. ``starts``/``ends``/``n_lines`` give the lines back; style
    keywords (``stroke_width``, ``stroke_color``, ``stroke_opacity``) go to
    ``VMobject``.
    """

    def __init__(self, starts: Any, ends: Any, **kwargs):
        super().__init__(**kwargs)
        a = np.asarray(starts, dtype=float).reshape(-1, 3)
        b = np.asarray(ends, dtype=float).reshape(-1, 3)
        if len(a):
            # as Line's set_points_as_corners: handles at 1/3 and 2/3 of each line
            n = self.n_points_per_cubic_curve
            pts = [(1 - t) * a + t * b for t in np.linspace(0, 1, n)]
            self.set_points(np.stack(pts, axis=1).reshape(-1, 3))

    @property
    def starts(self) -> np.ndarray:
        return self.points[:: self.n_points_per_cubic_curve].copy()

    @property
    def ends(self) -> np.ndarray:
        n = self.n_points_per_cubic_curve
        return self.points[n - 1 :: n].copy()

    @property
    def n_lines(self) -> int:
        return len(self.points) // self.n_points_per_cubic_curve


def grid_line_count(grid: VGroup) -> int:
    """Number of lines in a grid group (``panel.grid``, ``grid_major`` or
    ``grid_minor``) made of :class:`GridLines` (any other member counts as one)."""
    return sum(getattr(m, "n_lines", 1) for m in grid.submobjects)


#: Smallest gap between neighbouring tick labels, as a fraction of the height of
#: one line of axis text (:func:`fit_breaks`).
TICK_LABEL_GAP = 0.25


def fit_breaks(
    pp: PanelParams, x_scale: Any, y_scale: Any, width: float, height: float, theme: Theme
) -> PanelParams:
    """``pp`` with *automatic* breaks coarsened where their tick labels would not fit.

    ggplot2 picks breaks without looking at the panel size; ggmanim's panels can be
    small (facets), so for each axis whose scale has default breaks
    (``scale.has_default_breaks()``) the labels are measured (with
    :meth:`Theme.make_text`, rotated and justified as they will be drawn) and placed
    at their ticks on an axis ``width`` (x) / ``height`` (y) long. If two neighbours
    come closer than :data:`TICK_LABEL_GAP` lines of text, the breaks are recomputed
    from ``scale.with_n_breaks(n)`` for ``n = n_breaks - 1, ..., 2`` until they fit.
    When even the coarsest of those is crowded (``pretty()`` has no nicer step
    leaving two breaks in range), every k-th break is kept instead (k = 2, 3, ...;
    the roundest subset first). At least two breaks remain: if nothing fits, the
    coarsest set is used. Minor breaks follow the new majors.

    User-given breaks (sequence or callable) are never changed. Discrete axes are
    left alone too: their labels are the levels, so crowding there is left to the
    theme (``axis_text_x_angle``, a smaller ``axis_text_size``). So are
    :func:`~ggmanim.coords_trans.coord_trans` axes (their scale facade does not
    define ``with_n_breaks``).

    The labels are measured even when the theme hides them, so panels sharing a
    scale and an axis length (a facet column) get the same breaks.
    """
    xb, xl, xm = _fit_axis(
        x_scale, pp.x_range, pp.x_breaks, pp.x_labels, pp.x_minor_breaks, width, theme, "x"
    )
    yb, yl, ym = _fit_axis(
        y_scale, pp.y_range, pp.y_breaks, pp.y_labels, pp.y_minor_breaks, height, theme, "y"
    )
    if (xb, yb) == (pp.x_breaks, pp.y_breaks):
        return pp
    return replace(
        pp, x_breaks=xb, x_labels=xl, x_minor_breaks=xm, y_breaks=yb, y_labels=yl, y_minor_breaks=ym
    )


def _fit_axis(scale, rng, breaks, labels, minor, length: float, theme: Theme, axis: str):
    """``(breaks, labels, minor)`` for one axis; see :func:`fit_breaks`."""
    # looked up on the class: attribute-forwarding facades (coord_trans) opt out
    cls = type(scale)
    if not (hasattr(cls, "has_default_breaks") and hasattr(cls, "with_n_breaks")):
        return breaks, labels, minor
    if not scale.has_default_breaks() or len(breaks) < 3:
        return breaks, labels, minor
    far = _side(scale, "bottom" if axis == "x" else "left") in ("top", "right")
    if not _crowded(breaks, labels, rng, length, theme, axis, far):
        return breaks, labels, minor
    best = (breaks, labels, minor)
    for n in range(int(scale.n_breaks) - 1, 1, -1):
        thin = scale.with_n_breaks(n)
        b, lab = _breaks_labels(thin, rng)
        if len(b) < 2:
            break
        if len(b) >= len(best[0]):
            continue
        best = (b, lab, _minor_breaks(thin, rng))
        if not _crowded(b, lab, rng, length, theme, axis, far):
            return best
    # pretty() may have no coarser nice step that leaves two breaks inside the
    # range (1000..9000: step 2000 gives four, 5000 only one): keep every k-th
    b0, l0, m0 = best
    for k in range(2, len(b0)):
        for idx in _strides(b0, k):
            b = tuple(float(b0[i]) for i in idx)
            lab = tuple(l0[i] for i in idx)
            best = (b, lab, _thinned_minor(scale, b0, b, m0))
            if not _crowded(b, lab, rng, length, theme, axis, far):
                return best
    return best  # nothing fits: the coarsest set (two breaks)


def _strides(breaks, k: int) -> list[list[int]]:
    """Index sets ``breaks[off::k]`` with at least two members, the "roundest"
    first (most breaks that are multiples of ``k`` steps, e.g. 4000/8000 before
    2000/6000 for k=2)."""
    step = float(breaks[1]) - float(breaks[0])
    sets = [list(range(off, len(breaks), k)) for off in range(k)]
    sets = [s for s in sets if len(s) >= 2]

    def round_count(idx: list[int]) -> int:
        q = np.asarray([float(breaks[i]) for i in idx]) / (k * step)
        return int(np.sum(np.abs(q - np.round(q)) < 1e-6))

    return sorted(sets, key=lambda s: (-round_count(s), -len(s)))


def _thinned_minor(scale, old, new, minor) -> tuple[float, ...]:
    """Minor breaks after dropping majors: with the default minor breaks the
    dropped majors become minor ones (the grid keeps its density)."""
    if getattr(scale, "minor_breaks", ()) is not None:
        return tuple(minor)
    dropped = set(float(b) for b in old) - set(new)
    return tuple(sorted(set(minor) | dropped))


def _crowded(breaks, labels, rng, length: float, theme: Theme, axis: str, far: bool) -> bool:
    """Do two neighbouring tick labels come closer than :data:`TICK_LABEL_GAP` lines
    of text along the axis (labels as :func:`_place_tick_label` places them)?"""
    lo, hi = rng
    if hi <= lo or length <= 0:
        return False
    i = 0 if axis == "x" else 1
    if axis == "x":
        direction = UP if far else DOWN
    else:
        direction = RIGHT if far else LEFT
    spans = []
    for b, s in zip(breaks, labels):
        lab = _label(theme, s, axis)
        if len(lab.get_all_points()) == 0:
            continue
        _place_tick_label(lab, np.zeros(3), direction, 0.0, theme, axis)
        at = (float(b) - lo) / (hi - lo) * length
        spans.append((at + lab.get_corner(DL)[i], at + lab.get_corner(UR)[i]))
    if len(spans) < 2:
        return False
    gap = TICK_LABEL_GAP * _line_height(theme)
    spans.sort()
    return any(b[0] - a[1] < gap for a, b in zip(spans, spans[1:]))


def _line_height(theme: Theme) -> float:
    return float(theme.make_text("0", **theme.text_style("axis_text")).height)


# endregion


# region: cartesian
class CoordCartesian:
    """Cartesian coordinates.

    ``xlim``/``ylim`` (data space, optional) *zoom* the panel without touching the
    data: no values are censored or dropped. This happens *after* the stats, so a
    ``geom_smooth`` is fitted to all the data and its line runs to the panel edge.
    Scale limits (``xlim()``, ``scale_x_continuous(limits=)``) instead apply the
    scale's ``oob`` (default: censor) *before* the stats, so the fit only sees
    in-range data.

    ``expand=False`` turns off the scales' expansion (for zoomed and unzoomed axes).
    ``clip="on"`` (default) clips marks to the panel: points outside are dropped,
    paths/ribbons/bars are cut at the border (:mod:`ggmanim.geoms._clip`);
    ``clip="off"`` draws them wherever they fall.
    """

    def __init__(
        self,
        xlim: Sequence[float | None] | None = None,
        ylim: Sequence[float | None] | None = None,
        expand: bool = True,
        clip: str = "on",
    ):
        if clip not in _CLIP:
            raise ValueError(f"clip must be 'on' or 'off', got {clip!r}")
        self.clip = clip
        for name, lim in (("xlim", xlim), ("ylim", ylim)):
            if lim is not None and len(tuple(lim)) != 2:
                raise ValueError(f"{name} must be (lo, hi), got {lim!r}")
        self.xlim = None if xlim is None else tuple(xlim)
        self.ylim = None if ylim is None else tuple(ylim)
        self.expand = bool(expand)

    def add_to(self, plot):
        return plot._with_coord(self)

    def __repr__(self):
        clip = f", clip={self.clip!r}" if self.clip != "on" else ""
        return (
            f"coord_cartesian(xlim={self.xlim!r}, ylim={self.ylim!r}, expand={self.expand!r}{clip})"
        )

    # -- geometry --------------------------------------------------------------
    def _axis_range(self, scale, lim) -> tuple[float, float]:
        return _range(scale, lim, self.expand)

    def _panel_size(self, theme: Theme, has_legend: bool, pp: PanelParams) -> tuple[float, float]:
        """``(width, height)`` of the panel; :class:`CoordFixed` overrides this."""
        return theme.panel_size(has_legend)

    def setup_panel_params(self, x_scale, y_scale) -> PanelParams:
        """Ranges, breaks and labels for trained ``x_scale``/``y_scale`` (either may
        be ``None``: range ``(0, 1)``, no breaks). Does not mutate the scales."""
        xr = self._axis_range(x_scale, self.xlim)
        yr = self._axis_range(y_scale, self.ylim)
        xb, xl = _breaks_labels(x_scale, xr)
        yb, yl = _breaks_labels(y_scale, yr)
        return PanelParams(
            x_range=xr,
            y_range=yr,
            x_breaks=xb,
            y_breaks=yb,
            x_labels=xl,
            y_labels=yl,
            x_minor_breaks=_minor_breaks(x_scale, xr),
            y_minor_breaks=_minor_breaks(y_scale, yr),
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
        width, height = self._panel_size(theme, has_legend, pp)
        pp = fit_breaks(pp, x_scale, y_scale, width, height, theme)  # ggm-rnm
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
        _style_axis_lines(axes, theme)

        xb = np.asarray(pp.x_breaks, dtype=float)
        yb = np.asarray(pp.y_breaks, dtype=float)
        bottom = _c2p(axes, xb, y0)
        top = _c2p(axes, xb, y1)
        left = _c2p(axes, x0, yb)
        right = _c2p(axes, x1, yb)

        grid_major = _major_grid(theme, bottom, top, left, right)
        grid_minor = _minor_grid(theme, axes, pp)
        grid = VGroup(*grid_minor, *grid_major)

        # axis sides (scale position="top"/"right")
        # "top"/"right" is the far side; under coord_flip the scales arrive swapped,
        # so an x scale's "top" puts the axis on the right (as in ggplot2)
        x_top = _side(x_scale, "bottom") in ("top", "right")
        y_right = _side(y_scale, "left") in ("right", "top")
        x_at, x_dir = (top, UP) if x_top else (bottom, DOWN)
        y_at, y_dir = (right, RIGHT) if y_right else (left, LEFT)
        _move_axis_lines(axes, theme, x_top, y_right)

        tl = theme.tick_length
        x_ticks, y_ticks = VGroup(), VGroup()
        tick_color = theme.color_of("axis_ticks_color")
        if theme.axis_element("axis_ticks", "x"):
            for p in x_at:
                x_ticks.add(
                    Line(
                        p,
                        p + tl * x_dir,
                        stroke_width=theme.axis_stroke_width,
                        stroke_color=tick_color,
                    )
                )
        if theme.axis_element("axis_ticks", "y"):
            for p in y_at:
                y_ticks.add(
                    Line(
                        p,
                        p + tl * y_dir,
                        stroke_width=theme.axis_stroke_width,
                        stroke_color=tick_color,
                    )
                )

        x_gap = (tl if len(x_ticks) else 0.0) + theme.tick_label_buff
        y_gap = (tl if len(y_ticks) else 0.0) + theme.tick_label_buff
        x_tick_labels, y_tick_labels = VGroup(), VGroup()
        if theme.axis_element("axis_text", "x"):
            for p, s in zip(x_at, pp.x_labels):
                lab = _place_tick_label(_label(theme, s, "x"), p, x_dir, x_gap, theme, "x")
                x_tick_labels.add(lab)  # type: ignore[arg-type]
        if theme.axis_element("axis_text", "y"):
            for p, s in zip(y_at, pp.y_labels):
                lab = _place_tick_label(_label(theme, s, "y"), p, y_dir, y_gap, theme, "y")
                y_tick_labels.add(lab)  # type: ignore[arg-type]

        x_title = y_title = None
        if theme.axis_element("axis_title", "x"):
            x_title = _make_title(theme, _title(x_scale, labels, "x"))
        if theme.axis_element("axis_title", "y"):
            y_title = _make_title(theme, _title(y_scale, labels, "y"))
        corner = _c2p(axes, x1 if y_right else x0, y1 if x_top else y0)[0]
        centre = _c2p(axes, (x0 + x1) / 2, (y0 + y1) / 2)[0]
        if x_title is not None:
            below = (
                x_tick_labels
                if len(x_tick_labels)
                else VGroup(Line(corner, corner + x_gap * x_dir))
            )
            x_title.next_to(below, x_dir, buff=theme.axis_title_buff)
            x_title.set_x(centre[0])
        if y_title is not None:
            y_title.rotate(-PI / 2 if y_right else PI / 2)
            beside = (
                y_tick_labels
                if len(y_tick_labels)
                else VGroup(Line(corner, corner + y_gap * y_dir))
            )
            y_title.next_to(beside, y_dir, buff=theme.axis_title_buff)
            y_title.set_y(centre[1])

        panel = CartesianPanel(
            axes, pp, grid, x_ticks, y_ticks, x_tick_labels, y_tick_labels, x_title, y_title
        )
        panel.grid_major, panel.grid_minor = grid_major, grid_minor
        _add_secondary_axes(panel, theme, x_scale, y_scale, labels, x_top, y_right)
        _add_panel_decorations(panel, theme)
        panel.clip = self.clip
        return panel


# -- secondary axes (sec_axis / dup_axis) ----------------------------------------------
_SEC_PARTS = ("sec_line", "sec_ticks", "sec_tick_labels")


def _add_secondary_axes(
    panel: CartesianPanel,
    theme: Theme,
    x_scale,
    y_scale,
    labels: Mapping[str, Any] | None,
    x_top: bool,
    y_right: bool,
) -> None:
    """Draw each scale's ``sec_axis`` on the side opposite its primary axis.

    Sets, for the horizontal (``x_``) and vertical (``y_``) axis of the panel
    (screen orientation, like ``x_ticks``; under :func:`coord_flip` ``x_`` is the
    y aesthetic's): ``x_sec_breaks``/``x_sec_labels`` (positions on the drawn
    axis, i.e. the scale's transformed space, coord-transformed under
    :func:`~ggmanim.coords_trans.coord_trans`; and the label strings), ``x_sec_line`` (axis line or ``None``),
    ``x_sec_ticks``, ``x_sec_tick_labels`` (``VGroup``), ``x_sec_title``
    (``Mobject`` or ``None``) and ``x_sec_side``; plus ``panel.secondary_parts``,
    ``{side: [ticks, tick labels]}`` for the facet strip placement. Empty groups
    and ``None`` when the scale has no secondary axis."""
    axes = panel.axes
    (x0, x1), (y0, y1) = axes.x_range[:2], axes.y_range[:2]
    centre = _c2p(axes, (x0 + x1) / 2, (y0 + y1) / 2)[0]
    tl = theme.tick_length
    tick_color = theme.color_of("axis_ticks_color")
    line_color = theme.color_of("axis_line_color")
    panel.secondary_parts = {}  # type: ignore[attr-defined]
    for axis, scale, far in (("x", x_scale, x_top), ("y", y_scale, y_right)):
        sec = getattr(scale, "sec_axis", None) if scale is not None else None
        ticks, tick_labels = VGroup(), VGroup()
        line: Mobject | None = None
        title: Mobject | None = None
        breaks: tuple[float, ...] = ()
        strings: tuple[str, ...] = ()
        if axis == "x":
            side, direction = ("bottom", DOWN) if far else ("top", UP)
            edge = y0 if far else y1
        else:
            side, direction = ("left", LEFT) if far else ("right", RIGHT)
            edge = x0 if far else x1
        if sec is not None:
            breaks, strings = _sec_break_info(sec, scale, (x0, x1) if axis == "x" else (y0, y1))
            b = np.asarray(breaks, dtype=float)
            at = _c2p(axes, b, edge) if axis == "x" else _c2p(axes, edge, b)
            if theme.axis_element("axis_line", axis):
                ends = ((x0, edge), (x1, edge)) if axis == "x" else ((edge, y0), (edge, y1))
                line = Line(
                    _c2p(axes, *ends[0])[0],
                    _c2p(axes, *ends[1])[0],
                    stroke_width=theme.axis_stroke_width,
                    stroke_color=line_color,
                )
            if theme.axis_element("axis_ticks", axis):
                for p in at:
                    ticks.add(
                        Line(
                            p,
                            p + tl * direction,
                            stroke_width=theme.axis_stroke_width,
                            stroke_color=tick_color,
                        )
                    )
            gap = (tl if len(ticks) else 0.0) + theme.tick_label_buff
            if theme.axis_element("axis_text", axis):
                for p, s in zip(at, strings):
                    lab = _place_tick_label(_label(theme, s, axis), p, direction, gap, theme, axis)
                    tick_labels.add(lab)  # type: ignore[arg-type]
            if theme.axis_element("axis_title", axis):
                title = _make_title(theme, sec.title(_title(scale, labels, axis)))
            if title is not None:
                corner = _c2p(axes, *((x0, edge) if axis == "x" else (edge, y0)))[0]
                beside = (
                    tick_labels
                    if len(tick_labels)
                    else VGroup(Line(corner, corner + gap * direction))
                )
                if axis == "y":
                    title.rotate(PI / 2 if far else -PI / 2)
                title.next_to(beside, direction, buff=theme.axis_title_buff)
                if axis == "x":
                    title.set_x(centre[0])
                else:
                    title.set_y(centre[1])
            panel.secondary_parts[side] = [ticks, tick_labels]
            if line is not None:
                panel.add(line)  # type: ignore[arg-type]
            panel.add(ticks, tick_labels)
            if title is not None:
                panel.add(title)  # type: ignore[arg-type]
        setattr(panel, f"{axis}_sec_breaks", breaks)
        setattr(panel, f"{axis}_sec_labels", strings)
        setattr(panel, f"{axis}_sec_line", line)
        setattr(panel, f"{axis}_sec_ticks", ticks)
        setattr(panel, f"{axis}_sec_tick_labels", tick_labels)
        setattr(panel, f"{axis}_sec_title", title)
        setattr(panel, f"{axis}_sec_side", side if sec is not None else None)


def _sec_break_info(sec, scale, rng) -> tuple[tuple[float, ...], tuple[str, ...]]:
    """``sec.break_info`` over the drawn range ``rng``. A :func:`coord_trans` axis
    scale (a facade with ``input_range``) is drawn in coord-transformed space:
    the breaks are found over its input range and mapped through its transform."""
    inner = getattr(scale, "input_range", None)
    trans = getattr(scale, "_trans", None)
    if inner is None or trans is None:
        return sec.break_info(scale, rng)
    pos, strings = sec.break_info(scale, inner)
    with np.errstate(all="ignore"):
        t = np.asarray(trans.transform(np.asarray(pos, dtype=float)), dtype=float)
    keep = np.isfinite(t)
    return (
        tuple(float(v) for v in t[keep]),
        tuple(lab for lab, k in zip(strings, keep) if k),
    )


def _keep_axes(
    panel: Any, horizontal: bool, vertical: bool, sec_horizontal: bool, sec_vertical: bool
) -> None:
    """Facets: empty the tick marks / labels of the primary axes and drop the
    secondary axis line, ticks and labels that the facet does not draw (inner
    panels). Groups are emptied in place, so ``panel.x_ticks``,
    ``panel.x_sec_ticks`` etc. stay valid (and empty)."""
    for axis, keep in (("x", horizontal), ("y", vertical)):
        if not keep:
            for name in ("ticks", "tick_labels"):
                group = getattr(panel, f"{axis}_{name}")
                group.remove(*group.submobjects)
    for axis, keep in (("x", sec_horizontal), ("y", sec_vertical)):
        if keep:
            continue
        line = getattr(panel, f"{axis}_sec_line", None)
        if line is not None:
            panel.remove(line)
            setattr(panel, f"{axis}_sec_line", None)
        for name in ("sec_ticks", "sec_tick_labels"):
            group = getattr(panel, f"{axis}_{name}", None)
            if group is not None:
                group.remove(*group.submobjects)


def _secondary_title_content(scale, labels: Mapping[str, Any] | None, aes: str):
    """Title content of ``scale``'s secondary axis (``None``: no title / no axis)."""
    sec = getattr(scale, "sec_axis", None) if scale is not None else None
    if sec is None:
        return None
    return sec.title(_title(scale, labels, aes))


def _add_secondary_titles(
    grid: Any,
    h_scale,
    v_scale,
    h_aes: str,
    v_aes: str,
    labels: Mapping[str, Any] | None,
    theme: Theme,
    sides: tuple[str, str],
) -> None:
    """Facets: one secondary axis title per direction for the whole panel grid,
    outside it on the side opposite ``sides`` (the primary axis sides). Sets
    ``grid.x_sec_title`` / ``grid.y_sec_title`` (horizontal / vertical axis)."""
    lo = grid.axes.get_corner(DL)
    hi = grid.axes.get_corner(UR)
    grid.x_sec_title = grid.y_sec_title = None
    if theme.axis_element("axis_title", "x"):
        t = _make_title(theme, _secondary_title_content(h_scale, labels, h_aes))
        if t is not None:
            t.next_to(
                VGroup(*grid.submobjects),
                UP if sides[0] == "bottom" else DOWN,
                buff=theme.axis_title_buff,
            )
            t.set_x((lo[0] + hi[0]) / 2)
            grid.x_sec_title = t
    if theme.axis_element("axis_title", "y"):
        t = _make_title(theme, _secondary_title_content(v_scale, labels, v_aes))
        if t is not None:
            left = sides[1] == "right"
            t.rotate(PI / 2 if left else -PI / 2)
            t.next_to(
                VGroup(*grid.submobjects), LEFT if left else RIGHT, buff=theme.axis_title_buff
            )
            t.set_y((lo[1] + hi[1]) / 2)
            grid.y_sec_title = t
    for m in (grid.x_sec_title, grid.y_sec_title):
        if m is not None:
            grid.add(m)


def _label(theme: Theme, s: str, axis: str = "x") -> Mobject:
    if s == "":
        return VMobject()  # keep one label per break; nothing to draw
    text = render_label(s, theme.text_engine)
    if theme.text_engine == "text" and "\n" in text:
        return _multiline_label(theme, text, axis)
    return theme.make_text(text, **theme.text_style("axis_text"))


def _multiline_label(theme: Theme, text: str, axis: str) -> Mobject:
    """A multi-line tick label (e.g. from :func:`~ggmanim.labels.label_date_short`):
    a Pango :class:`~manim.Paragraph` (via :meth:`Theme.make_text`) whose lines are
    justified by the axis text ``hjust`` (x labels centred, y labels right-aligned by
    default), as in ggplot2."""
    style = theme.text_style("axis_text")
    hjust = theme.axis_text_just(axis)[0]
    return theme.make_text(text, align=hjust_align(hjust), **style)


# -- theme elements (backgrounds, borders, grid, axis lines, rotated labels) ----------
def _style_axis_lines(axes: Axes, theme: Theme) -> None:
    """Colour/width the axis lines; hide the ones switched off (per axis)."""
    color = theme.color_of("axis_line_color")
    for axis, line in (("x", axes.x_axis), ("y", axes.y_axis)):
        on = theme.axis_element("axis_line", axis)
        line.set_stroke(color=color, width=theme.axis_stroke_width, opacity=1.0 if on else 0.0)


def _major_grid(theme: Theme, bottom, top, left, right) -> VGroup:
    """Major grid lines: vertical ones at the x breaks, horizontal at the y breaks
    (one :class:`GridLines` per axis)."""
    lines = []
    if theme.grid_major("x"):
        lines.append((bottom, top))
    if theme.grid_major("y"):
        lines.append((left, right))
    style = {
        "stroke_width": theme.grid_stroke_width,
        "stroke_color": theme.color_of("grid_color"),
        "stroke_opacity": theme.grid_opacity,
    }
    return VGroup(*(GridLines(a, b, **style) for a, b in lines if len(a)))


def _minor_grid(theme: Theme, axes: Axes, pp: PanelParams) -> VGroup:
    """Minor grid lines at the scales' minor breaks (``theme.panel_grid_minor``), one
    :class:`GridLines` per axis."""
    if not getattr(theme, "panel_grid_minor", True):
        return VGroup()
    (x0, x1), (y0, y1) = pp.x_range, pp.y_range
    xm = np.asarray(pp.x_minor_breaks, dtype=float)
    ym = np.asarray(pp.y_minor_breaks, dtype=float)
    lines = [(_c2p(axes, xm, y0), _c2p(axes, xm, y1)), (_c2p(axes, x0, ym), _c2p(axes, x1, ym))]
    style = {
        "stroke_width": theme.grid_minor_stroke_width,
        "stroke_color": theme.color_of("grid_color"),
        "stroke_opacity": theme.grid_minor_opacity,
    }
    return VGroup(*(GridLines(a, b, **style) for a, b in lines if len(a)))


def _move_axis_lines(axes: Axes, theme: Theme, x_top: bool, y_right: bool) -> None:
    """For ``position="top"``/``"right"`` scales: hide manim's axis line (which must
    stay put so ``c2p`` is unchanged) and draw one along the opposite panel edge."""
    (x0, x1), (y0, y1) = axes.x_range[:2], axes.y_range[:2]
    color = theme.color_of("axis_line_color")
    for moved, axis, line, a, b in (
        (x_top, "x", axes.x_axis, (x0, y1), (x1, y1)),
        (y_right, "y", axes.y_axis, (x1, y0), (x1, y1)),
    ):
        if not moved:
            continue
        line.set_stroke(opacity=0.0)
        if theme.axis_element("axis_line", axis):
            axes.add(
                Line(
                    _c2p(axes, *a)[0],
                    _c2p(axes, *b)[0],
                    stroke_width=theme.axis_stroke_width,
                    stroke_color=color,
                )
            )


def _place_tick_label(
    label: Mobject, point: np.ndarray, direction: np.ndarray, gap: float, theme: Theme, axis: str
) -> Mobject:
    """Rotate ``label`` by ``axis_text_{axis}_angle`` and pin its (hjust, vjust)
    anchor at ``gap`` from ``point`` in ``direction`` (ggplot semantics: the
    anchor is a point of the *unrotated* label box). With automatic justification
    the label is then nudged so no part of it lies on the axis side of the anchor.
    With angle 0 and automatic justification this equals ``next_to(point, direction, gap)``."""
    target = np.asarray(point, dtype=float) + gap * np.asarray(direction, dtype=float)
    if len(label.get_all_points()) == 0:
        return label.move_to(target)
    angle = getattr(theme, f"axis_text_{axis}_angle")
    hjust, vjust, auto = theme.axis_text_just(axis)
    if auto and float(np.dot(direction, UP + RIGHT)) > 0:
        # automatic justification is for bottom/left axes: mirror it for top/right
        hjust, vjust = 1.0 - hjust, 1.0 - vjust
    lo, hi = label.get_corner(DL), label.get_corner(UR)
    anchor = np.array([lo[0] + hjust * (hi[0] - lo[0]), lo[1] + vjust * (hi[1] - lo[1]), 0.0])
    if angle:
        label.rotate(angle * DEGREES, about_point=anchor)
    label.shift(target - anchor)
    if auto:
        back = -np.asarray(direction, dtype=float)
        overshoot = float(np.dot(label.get_critical_point(back) - target, back))
        if overshoot > 1e-9:
            label.shift(-overshoot * back)
    return label


def _add_panel_decorations(panel: CartesianPanel, theme: Theme) -> None:
    """Add the panel background (behind everything) and border (in front of the
    axes) as ``panel.background`` / ``panel.border`` when the theme asks for them."""
    (x0, x1), (y0, y1) = panel.panel_params.x_range, panel.panel_params.y_range
    lo = _c2p(panel.axes, x0, y0)[0]
    hi = _c2p(panel.axes, x1, y1)[0]
    if theme.panel_background is not None:
        bg = Polygon(*_box(lo, hi), stroke_width=0)
        bg.set_fill(ManimColor(theme.panel_background), opacity=theme.panel_background_opacity)
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


def _box(lo: np.ndarray, hi: np.ndarray) -> list[np.ndarray]:
    return [
        np.array([lo[0], lo[1], 0.0]),
        np.array([hi[0], lo[1], 0.0]),
        np.array([hi[0], hi[1], 0.0]),
        np.array([lo[0], hi[1], 0.0]),
    ]


def _make_title(theme: Theme, content: Any) -> Mobject | None:
    if content is None:
        return None
    if isinstance(content, Mobject):
        return content.copy()  # never move/rotate the caller's mobject
    if str(content) == "":
        return None
    return theme.make_text(content, **theme.text_style("axis_title"))


@register
def coord_cartesian(xlim=None, ylim=None, expand: bool = True, clip: str = "on") -> CoordCartesian:
    """Cartesian coordinates; ``xlim``/``ylim`` zoom without censoring data and
    ``clip="on"`` keeps marks inside the panel (see :class:`CoordCartesian`)."""
    return CoordCartesian(xlim=xlim, ylim=ylim, expand=expand, clip=clip)


def _swap_xy(labels: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if labels is None:
        return None
    out = {k: v for k, v in labels.items() if k not in ("x", "y")}
    if "y" in labels:
        out["x"] = labels["y"]
    if "x" in labels:
        out["y"] = labels["x"]
    return out


class CoordFlip(CoordCartesian):
    """Cartesian coordinates with x and y swapped: the x aesthetic (and its scale,
    breaks and title) is drawn on the vertical axis, y on the horizontal one.

    Geoms are unchanged: they go through ``panel.transform(x, y)``, which swaps.
    ``xlim``/``ylim`` refer to the x/y *aesthetics* (as in ggplot2).
    ``panel.panel_params`` stays in aesthetic orientation (see
    :class:`CartesianPanel`).
    """

    def __repr__(self):
        return f"coord_flip(xlim={self.xlim!r}, ylim={self.ylim!r}, expand={self.expand!r})"

    def build(
        self,
        x_scale,
        y_scale,
        theme: Theme | None = None,
        labels: Mapping[str, Any] | None = None,
        *,
        has_legend: bool = False,
    ) -> CartesianPanel:
        inner = CoordCartesian(xlim=self.ylim, ylim=self.xlim, expand=self.expand, clip=self.clip)
        panel = inner.build(y_scale, x_scale, theme, _swap_xy(labels), has_legend=has_legend)
        pp = panel.panel_params
        panel.panel_params = PanelParams(
            x_range=pp.y_range,
            y_range=pp.x_range,
            x_breaks=pp.y_breaks,
            y_breaks=pp.x_breaks,
            x_labels=pp.y_labels,
            y_labels=pp.x_labels,
            x_minor_breaks=pp.y_minor_breaks,
            y_minor_breaks=pp.x_minor_breaks,
            flipped=True,
        )
        panel.flipped = True
        return panel


class CoordFixed(CoordCartesian):
    """Cartesian coordinates with a fixed aspect ratio: one unit on the y axis is
    ``ratio`` times as long on screen as one unit on the x axis (in transformed
    space). The panel is the largest such rectangle inside ``theme.panel_size``.
    """

    def __init__(
        self,
        ratio: float = 1.0,
        xlim: Sequence[float | None] | None = None,
        ylim: Sequence[float | None] | None = None,
        expand: bool = True,
        clip: str = "on",
    ):
        super().__init__(xlim=xlim, ylim=ylim, expand=expand, clip=clip)
        ratio = float(ratio)
        if not (np.isfinite(ratio) and ratio > 0):
            raise ValueError(f"ratio must be a positive number, got {ratio!r}")
        self.ratio = ratio

    def __repr__(self):
        return (
            f"coord_fixed(ratio={self.ratio!r}, xlim={self.xlim!r}, ylim={self.ylim!r}, "
            f"expand={self.expand!r})"
        )

    def _panel_size(self, theme: Theme, has_legend: bool, pp: PanelParams) -> tuple[float, float]:
        w, h = theme.panel_size(has_legend)
        dx = pp.x_range[1] - pp.x_range[0]
        dy = pp.y_range[1] - pp.y_range[0]
        aspect = self.ratio * dy / dx  # wanted height / width
        if aspect > h / w:
            return h / aspect, h
        return w, w * aspect


@register
def coord_flip(xlim=None, ylim=None, expand: bool = True, clip: str = "on") -> CoordFlip:
    """Flip x and y: the x aesthetic runs vertically (horizontal bars, etc.)."""
    return CoordFlip(xlim=xlim, ylim=ylim, expand=expand, clip=clip)


@register
def coord_fixed(
    ratio: float = 1.0, xlim=None, ylim=None, expand: bool = True, clip: str = "on"
) -> CoordFixed:
    """Fixed aspect ratio: ``ratio`` = screen length of one y unit / one x unit
    (``ratio=1``: a unit square is drawn square)."""
    return CoordFixed(ratio=ratio, xlim=xlim, ylim=ylim, expand=expand, clip=clip)


# endregion

# region: nonlinear
#: Munching resolution: a path segment is cut into pieces no longer than
#: ``1 / MUNCH_SEGMENTS`` of the panel (in the coord's normalised output space).
MUNCH_SEGMENTS = 50
#: Upper bound on the pieces one segment is cut into.
_MUNCH_MAX = 1000


def is_linear(coord: Any) -> bool:
    """``coord.is_linear`` (default ``True``): whether straight lines in the coord's
    input space stay straight on screen. Works on coord specs and built panels.

    Linear coords (cartesian, flip, fixed) map input space affinely to the scene,
    so geoms transform only the vertices; non-linear ones
    (:func:`~ggmanim.coords_trans.coord_trans`)
    need paths and polygons *munched* (:func:`coord_munch`) first."""
    return bool(getattr(coord, "is_linear", True))


def _munch_distance(coord: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Length of each segment of the path ``(x, y)`` as a fraction of the panel:
    ``coord.munch_distance(x, y)`` when the coord defines it, else the scene
    distance with each axis divided by the panel's scene width/height."""
    dist = getattr(coord, "munch_distance", None)
    if dist is not None:
        return np.asarray(dist(x, y), dtype=float)
    pts = np.asarray(coord.transform(x, y), dtype=float).reshape(-1, 3)
    pp = getattr(coord, "panel_params", None)
    w = h = 1.0
    if pp is not None:
        c = np.asarray(coord.transform(list(pp.x_range), list(pp.y_range)), dtype=float)
        w = abs(c[1, 0] - c[0, 0]) or 1.0
        h = abs(c[1, 1] - c[0, 1]) or 1.0
    d = np.diff(pts[:, :2], axis=0) / np.array([w, h])
    return np.hypot(d[:, 0], d[:, 1])


def coord_munch(
    coord: Any, x: Any, y: Any, *, closed: bool = False, segments: int = MUNCH_SEGMENTS
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the path ``(x, y)`` (coord *input* space) so that it draws as a
    curve under a non-linear coord (ggplot2's ``coord_munch``).

    Each segment is cut into ``ceil(length * segments)`` equal pieces (``length``
    measured in the coord's output space as a fraction of the panel, see
    ``coord.munch_distance``); the new vertices are placed *linearly in input
    space*, so after ``coord.transform`` they follow the image of the straight
    segment. ``closed=True`` also munches the closing edge (polygons); the result
    is still a ring without the first vertex repeated.

    Returns float arrays ``(x, y)``: unchanged (copies) when the coord is linear
    or the path has fewer than two vertices. Segments with a non-finite end are
    not subdivided, so NaN path breaks survive.
    """
    xs, ys = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    xs, ys = xs.ravel(), ys.ravel()
    if is_linear(coord) or len(xs) < 2:
        return xs.copy(), ys.copy()
    if closed:
        xs, ys = np.append(xs, xs[0]), np.append(ys, ys[0])
    with np.errstate(invalid="ignore"):
        d = _munch_distance(coord, xs, ys) * segments
        n = np.where(np.isfinite(d), np.clip(np.ceil(d), 1, _MUNCH_MAX), 1).astype(int)
    seg = np.repeat(np.arange(len(n)), n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n, n)
    with np.errstate(invalid="ignore"):
        mx = np.where(t == 0, xs[seg], xs[seg] + (xs[seg + 1] - xs[seg]) * t)
        my = np.where(t == 0, ys[seg], ys[seg] + (ys[seg + 1] - ys[seg]) * t)
    if closed:
        return mx, my
    return np.append(mx, xs[-1]), np.append(my, ys[-1])


# endregion
