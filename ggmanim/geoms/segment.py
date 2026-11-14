"""geom_segment / geom_curve: straight or curved lines from ``(x, y)`` to
``(xend, yend)``, optionally with arrowheads (:func:`arrow`)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from manim import LEFT, RIGHT, Line, LineJointType, Polygon, VGroup, VMobject

from ..registry import register
from ..theme import Theme, theme_default
from ._batch import draw_segments_fast, use_fast
from ._clip import clip_polyline, clip_segments, inside, panel_rect
from ._linetype import apply_linetype
from ._marker_clip import scene_box
from .base import (
    KEY_SIZE,
    Geom,
    attach_marks,
    finite_rows,
    make_layer,
    multi_path_mobject,
    num,
    path_mobject,
    resolve_color,
    transform_points,
)

__all__ = ["ArrowSpec", "arrow", "GeomSegment", "GeomCurve", "geom_segment", "geom_curve"]
__toplevel__ = ["arrow"]

_ENDS = ("last", "first", "both")
_TYPES = ("open", "closed")
#: Samples along a clipped curve.
_CURVE_SAMPLES = 64


@dataclass(frozen=True)
class ArrowSpec:
    """Arrowhead spec (see :func:`arrow`)."""

    angle: float = 30.0
    length: float = 0.15
    ends: str = "last"
    type: str = "open"

    def __post_init__(self):
        if self.ends not in _ENDS:
            raise ValueError(f"arrow ends must be one of {_ENDS}, got {self.ends!r}")
        if self.type not in _TYPES:
            raise ValueError(f"arrow type must be one of {_TYPES}, got {self.type!r}")
        if not (0 < float(self.angle) < 90):
            raise ValueError(f"arrow angle must be in (0, 90) degrees, got {self.angle!r}")
        if not float(self.length) > 0:
            raise ValueError(f"arrow length must be positive, got {self.length!r}")


def arrow(
    angle: float = 30, length: float = 0.15, ends: str = "last", type: str = "open"
) -> ArrowSpec:
    """Arrowhead specification for ``geom_segment(arrow=...)`` / ``geom_curve``.

    Parameters
    ----------
    angle : float
        Half-angle of the head in degrees (ggplot2/grid convention).
    length : float
        Length of the head's sides in scene units (ggplot2: ``unit(0.25, "inches")``).
    ends : ``"last"``, ``"first"`` or ``"both"``
        Which end(s) get a head.
    type : ``"open"`` or ``"closed"``
        Two strokes (a "V") or a filled triangle.
    """
    return ArrowSpec(angle=float(angle), length=float(length), ends=ends, type=type)


def _head(tip: np.ndarray, direction: np.ndarray, spec: ArrowSpec, color, width, alpha) -> VMobject:
    """An arrowhead with its point at ``tip`` pointing along ``direction``."""
    d = np.asarray(direction, dtype=float)
    n = float(np.linalg.norm(d[:2]))
    d = d / n if n > 0 else np.array([1.0, 0.0, 0.0])
    a = np.deg2rad(spec.angle)
    back = -d * spec.length
    c, s = np.cos(a), np.sin(a)
    left = tip + np.array([c * back[0] - s * back[1], s * back[0] + c * back[1], 0.0])
    right = tip + np.array([c * back[0] + s * back[1], -s * back[0] + c * back[1], 0.0])
    if spec.type == "closed":
        head: VMobject = Polygon(left, tip, right)
        head.set_fill(color, opacity=alpha)
        head.set_stroke(color=color, width=width, opacity=alpha)
        head.joint_type = LineJointType.MITER
    else:
        head = path_mobject(np.array([left, tip, right]), color, width, alpha)
    return head


def _with_heads(
    line: VMobject, spec: ArrowSpec | None, start, start_dir, end, end_dir, color, width, alpha
) -> VMobject:
    """``line`` or ``VGroup(line, *heads)``. ``start``/``end`` are ``None`` when
    that end was clipped away (no head there); ``*_dir`` point *outwards*."""
    if spec is None:
        return line
    heads = []
    if spec.ends in ("first", "both") and start is not None:
        heads.append(_head(start, start_dir, spec, color, width, alpha))
    if spec.ends in ("last", "both") and end is not None:
        heads.append(_head(end, end_dir, spec, color, width, alpha))
    return VGroup(line, *heads) if heads else line


class GeomSegment(Geom):
    """Straight segments from ``(x, y)`` to ``(xend, yend)``: one mark per
    visible row, the line (a ``VMobject``; dashes ``VGroup`` for a non-solid
    ``linetype``) or ``VGroup(line, *heads)`` with ``arrow=``. Clipped to the
    panel in transformed space; an end cut off by the panel gets no head.

    ``fast`` (``None``: above ``theme.fast_threshold`` rows) draws batched marks
    without arrowheads, one ``VMobject`` per style (:mod:`ggmanim.geoms._batch`;
    no per-mark ``.keys``); with ``arrow=`` every row is drawn individually."""

    required_aes = ("x", "y", "xend", "yend")
    default_aes = {"color": None, "linewidth": 2.0, "alpha": 1.0, "linetype": "solid"}
    extra_params: tuple[str, ...] = ("arrow", "lineend", "fast")

    def draw_panel(
        self,
        data,
        panel_params,
        coord,
        theme: Theme | None = None,
        arrow=None,
        *,
        fast=None,
        **params,
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        self.check_required_aes(data)
        data = finite_rows(self.use_defaults(data, params), self.required_aes)
        if arrow is not None:
            if fast:
                warnings.warn(
                    f"{type(self).__name__}: fast=True is ignored with arrow= "
                    "(arrowheads are drawn per mark)",
                    stacklevel=2,
                )
        elif use_fast(fast, data, theme, type(self).__name__):
            return draw_segments_fast(
                data,
                panel_params,
                coord,
                theme,
                lambda rows: self.draw_panel(
                    rows, panel_params, coord, theme, fast=False, **params
                ),
            )
        fg = theme.get_foreground()
        x0, y0 = data["x"].to_numpy(float), data["y"].to_numpy(float)
        x1, y1 = data["xend"].to_numpy(float), data["yend"].to_numpy(float)
        rect = panel_rect(coord, panel_params)
        if rect is None:
            t0, t1 = np.zeros(len(x0)), np.ones(len(x0))
            vis = np.ones(len(x0), dtype=bool)
        else:
            t0, t1, vis = clip_segments(x0, y0, x1, y1, rect)
        a = transform_points(coord, x0 + t0 * (x1 - x0), y0 + t0 * (y1 - y0))
        b = transform_points(coord, x0 + t1 * (x1 - x0), y0 + t1 * (y1 - y0))
        out, drawn = VGroup(), []
        for i, row in enumerate(data.to_dict("records")):
            if not vis[i] or np.allclose(a[i], b[i]):
                continue  # clipped away, or zero length (no direction for a head)
            color = resolve_color(row.get("color"), fg)
            width = num(row.get("linewidth"), 2.0)
            alpha = num(row.get("alpha"), 1.0)
            line: VMobject = Line(a[i], b[i], color=color, stroke_width=width, stroke_opacity=alpha)
            line = apply_linetype(line, row.get("linetype"))
            d = b[i] - a[i]
            mob = _with_heads(
                line,
                arrow,
                a[i] if t0[i] <= 0 else None,
                -d,
                b[i] if t1[i] >= 1 else None,
                d,
                color,
                width,
                alpha,
            )
            out.add(mob)
            drawn.append(i)
        return attach_marks(out, data.iloc[drawn], per="row")

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        row = self.key_aes(data_row)
        w = getattr(theme, "legend_key_width", None) or KEY_SIZE
        line = Line(
            LEFT * w / 2,
            RIGHT * w / 2,
            color=resolve_color(row.get("color"), theme.get_foreground()),
            stroke_width=num(row.get("linewidth"), 2.0),
            stroke_opacity=num(row.get("alpha"), 1.0),
        )
        return apply_linetype(line, row.get("linetype"), max_period=w / 2)


# -- grid's curveGrob ------------------------------------------------------------------
#
# ggplot2's geom_curve draws grid::curveGrob(curvature, angle, ncp, square = FALSE,
# shape = 0.5): ``ncp`` control points are placed by grid's calcControlPoints and an
# open X-spline (shape 0 at both ends, 0.5 at the control points) is drawn through
# them. The functions below port that construction (grid/R/curve.R, xspline.c) and
# run it in scene units (grid works in inches; both are isotropic and y-up).


def _curve_origin(x1, y1, x2, y2, origin):
    """grid's calcOrigin: the centre of rotation for the control points."""
    xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
    return xm + origin * (y2 - y1) / 2, ym - origin * (x2 - x1) / 2


def curve_control_points(
    start, end, curvature: float, angle: float | None = 90.0, ncp: int = 5
) -> np.ndarray:
    """``(ncp, 2)`` control points of grid's ``curveGrob`` between ``start`` and
    ``end`` (2-D, isotropic units). ``angle`` (degrees, 0..180) skews the curve:
    ``< 90`` towards the start, ``> 90`` towards the end; ``None`` picks it from the
    slope as grid does. Port of ``grid:::calcControlPoints``."""
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])
    xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    with np.errstate(all="ignore"):
        if angle is None:
            slope = np.float64(dy) / dx
            a = 2 * np.arctan(abs(slope)) if slope < 0 else 2 * np.arctan(1 / slope)
        else:
            a = np.deg2rad(angle)
        sina, cosa = np.sin(a), np.cos(a)
        cornerx = xm + (x1 - xm) * cosa - (y1 - ym) * sina
        cornery = ym + (y1 - ym) * cosa + (x1 - xm) * sina
        beta = -np.arctan(np.float64(cornery - y1) / (cornerx - x1))
        sinb, cosb = np.sin(beta), np.cos(beta)
        newx2 = x1 + dx * cosb - dy * sinb
        newy2 = y1 + dy * cosb + dx * sinb
        scalex = np.float64(newy2 - y1) / (newx2 - x1)
        newx1, newx2 = x1 * scalex, newx2 * scalex
        ratio = 2 * np.sin(np.arctan(curvature)) ** 2
        origin = curvature - curvature / ratio
        direction = 1.0 if curvature > 0 else -1.0
        ox, oy = _curve_origin(newx1, y1, newx2, newy2, origin)
        maxtheta = np.pi + np.sign(origin * direction) * 2 * np.arctan(abs(origin))
        theta = np.linspace(0, direction * maxtheta, int(ncp) + 2)[1:-1]
        cpx = ox + (newx1 - ox) * np.cos(theta) - (y1 - oy) * np.sin(theta)
        cpy = oy + (y1 - oy) * np.cos(theta) + (newx1 - ox) * np.sin(theta)
        cpx = cpx / scalex
        sinnb, cosnb = np.sin(-beta), np.cos(-beta)
        fx = x1 + (cpx - x1) * cosnb - (cpy - y1) * sinnb
        fy = y1 + (cpy - y1) * cosnb + (cpx - x1) * sinnb
    return np.column_stack([fx, fy])


def _f_blend(num: np.ndarray, den: float) -> np.ndarray:
    p = 2 * den * den
    u = num / den
    return u**3 * (10 - p + (2 * p - 15) * u + (6 - p) * u * u)


def xspline_open(points: np.ndarray, shapes, samples: int = 16) -> np.ndarray:
    """Points along the open X-spline (Blanc & Schlick, as in ``grid.xspline`` with
    ``repEnds=TRUE``) through ``points`` (``(n, d)``) with shapes in ``[0, 1]``
    (``0`` = pass through the control point, ``1`` = B-spline-like)."""
    pts = np.asarray(points, dtype=float)
    shp = np.asarray(shapes, dtype=float)
    if len(pts) < 2:
        return pts.copy()
    q = np.concatenate([pts[:1], pts, pts[-1:]])
    qs = np.concatenate([shp[:1], shp, shp[-1:]])
    t = np.arange(samples) / samples
    out = []
    for j in range(len(q) - 3):
        s1, s2 = qs[j + 1], qs[j + 2]
        a0 = np.where(t < s1, _f_blend(t - s1, -1 - s1), 0.0)
        a2 = _f_blend(t + s1, 1 + s1)
        a1 = _f_blend(t - 1 - s2, -1 - s2)
        a3 = np.where(t > 1 - s2, _f_blend(t - 1 + s2, 1 + s2), 0.0)
        w = np.column_stack([a0, a1, a2, a3])
        w = w / w.sum(axis=1, keepdims=True)
        out.append(w @ q[j : j + 4])
    out.append(pts[-1:])
    return np.concatenate(out)


def curve_points(
    start, end, curvature: float, angle: float | None = 90.0, ncp: int = 5
) -> np.ndarray:
    """The polyline (``(m, 3)``) ggplot2/grid draw for ``geom_curve`` from
    ``start`` to ``end`` (scene points): a straight segment for ``curvature == 0``
    or ``angle`` outside ``[1, 179]`` degrees (grid's rule), else the X-spline
    through :func:`curve_control_points`."""
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    straight = np.vstack([a, b])
    if curvature == 0 or (angle is not None and not 1 <= angle <= 179) or int(ncp) < 1:
        return straight
    cps = curve_control_points(a, b, curvature, angle, ncp)
    if not np.isfinite(cps).all():
        return straight
    ctrl = np.vstack([a[:2], cps, b[:2]])
    xy = xspline_open(ctrl, [0.0, *([0.5] * int(ncp)), 0.0])
    return np.column_stack([xy, np.full(len(xy), a[2] if len(a) > 2 else 0.0)])


class GeomCurve(GeomSegment):
    """Curved segments from ``(x, y)`` to ``(xend, yend)`` shaped like ggplot2's
    (grid's ``curveGrob``, see :func:`curve_points`): computed in scene space
    after the coord transform, so the shape does not depend on the aspect ratio of
    the axes. ``curvature`` (default ``0.5``): ``1`` is about a half circle,
    positive values bend to the right of the direction of travel, negative to the
    left, ``0`` is straight. ``angle`` (degrees, default ``90`` = symmetric) skews
    the curve towards the start (``< 90``) or the end (``> 90``); ``ncp`` (default
    ``5``) is the number of control points -- more gives a rounder curve. Each
    mark is a polyline, clipped to the panel in scene space."""

    extra_params: tuple[str, ...] = ("arrow", "lineend", "curvature", "angle", "ncp")

    def draw_panel(
        self,
        data,
        panel_params,
        coord,
        theme: Theme | None = None,
        arrow=None,
        curvature: float = 0.5,
        angle: float | None = 90.0,
        ncp: int = 5,
        **params,
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        self.check_required_aes(data)
        data = finite_rows(self.use_defaults(data, params), self.required_aes)
        fg = theme.get_foreground()
        a = transform_points(coord, data["x"].to_numpy(float), data["y"].to_numpy(float))
        b = transform_points(coord, data["xend"].to_numpy(float), data["yend"].to_numpy(float))
        rect = panel_rect(coord, panel_params)
        box = scene_box(coord, rect) if rect is not None else None
        out, drawn = VGroup(), []
        for i, row in enumerate(data.to_dict("records")):
            color: Any = resolve_color(row.get("color"), fg)
            width = num(row.get("linewidth"), 2.0)
            alpha = num(row.get("alpha"), 1.0)
            if np.allclose(a[i], b[i]):
                continue
            pts = curve_points(a[i], b[i], float(curvature), angle, int(ncp))
            # outward tangents at both ends
            start_dir, end_dir = pts[0] - pts[1], pts[-1] - pts[-2]
            start, end = a[i], b[i]
            pieces = [pts]
            if box is not None and not inside(pts[:, 0], pts[:, 1], box).all():
                if len(pts) == 2:  # straight: sample so the clip keeps the line
                    pts = a[i] + np.linspace(0, 1, _CURVE_SAMPLES + 1)[:, None] * (b[i] - a[i])
                clipped = clip_polyline(pts[:, 0], pts[:, 1], box)
                if not clipped:
                    continue
                z = float(a[i][2])
                pieces = [np.column_stack([px, py, np.full(len(px), z)]) for px, py in clipped]
                start = start if inside(pts[0, 0], pts[0, 1], box)[()] else None
                end = end if inside(pts[-1, 0], pts[-1, 1], box)[()] else None
            path = multi_path_mobject(pieces, color, width, alpha)
            line = apply_linetype(path, row.get("linetype"))
            out.add(_with_heads(line, arrow, start, start_dir, end, end_dir, color, width, alpha))
            drawn.append(i)
        return attach_marks(out, data.iloc[drawn], per="row")


def _layer(geom, mapping, data, stat, position, inherit_aes, name, params, arrow_spec):
    if arrow_spec is not None and not isinstance(arrow_spec, ArrowSpec):
        raise TypeError(f"arrow= must be made with arrow(), got {arrow_spec!r}")
    params["arrow"] = arrow_spec
    return make_layer(
        geom,
        mapping,
        data,
        stat=stat,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )


@register
def geom_segment(
    mapping=None,
    data=None,
    *,
    arrow: ArrowSpec | None = None,
    stat="identity",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params: Any,
):
    """Straight line segments from ``(x, y)`` to ``(xend, yend)``.

    ``arrow=arrow(...)`` adds arrowheads (sizes in scene units). ``lineend`` is
    accepted for ggplot2 compatibility (manim strokes have round ends).
    ``fast=True`` draws batched marks (default ``None``: when the panel has more
    than ``theme.fast_threshold`` rows), see :mod:`ggmanim.geoms._batch`.
    """
    return _layer(GeomSegment(), mapping, data, stat, position, inherit_aes, name, params, arrow)


@register
def geom_curve(
    mapping=None,
    data=None,
    *,
    curvature: float = 0.5,
    angle: float | None = 90.0,
    ncp: int = 5,
    arrow: ArrowSpec | None = None,
    stat="identity",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params: Any,
):
    """Curved line segments from ``(x, y)`` to ``(xend, yend)``, shaped like
    ggplot2's (grid's ``curveGrob``): ``curvature`` (``1`` ~ a half circle;
    negative values bend left, ``0`` is straight), ``angle`` (degrees in
    ``[0, 180]``; ``90`` is symmetric, smaller skews the curve towards the start)
    and ``ncp`` (number of control points). ``arrow=`` as for
    :func:`geom_segment`. See :class:`GeomCurve`."""
    if angle is not None and not 0 <= float(angle) <= 180:
        raise ValueError(f"geom_curve: angle must be in [0, 180] degrees, got {angle!r}")
    if int(ncp) < 1:
        raise ValueError(f"geom_curve: ncp must be >= 1, got {ncp!r}")
    params["curvature"] = float(curvature)
    params["angle"] = None if angle is None else float(angle)
    params["ncp"] = int(ncp)
    return _layer(GeomCurve(), mapping, data, stat, position, inherit_aes, name, params, arrow)
