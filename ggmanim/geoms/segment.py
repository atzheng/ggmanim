"""geom_segment / geom_curve: straight or curved lines from ``(x, y)`` to
``(xend, yend)``, optionally with arrowheads (:func:`arrow`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from manim import LEFT, PI, RIGHT, ArcBetweenPoints, Line, LineJointType, Polygon, VGroup, VMobject

from ..registry import register
from ..theme import Theme, theme_default
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
    panel in transformed space; an end cut off by the panel gets no head."""

    required_aes = ("x", "y", "xend", "yend")
    default_aes = {"color": None, "linewidth": 2.0, "alpha": 1.0, "linetype": "solid"}
    extra_params: tuple[str, ...] = ("arrow", "lineend")

    def draw_panel(
        self, data, panel_params, coord, theme: Theme | None = None, arrow=None, **params
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        self.check_required_aes(data)
        data = finite_rows(self.use_defaults(data, params), self.required_aes)
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


class GeomCurve(GeomSegment):
    """Curved segments: an ``ArcBetweenPoints`` from ``(x, y)`` to
    ``(xend, yend)`` (transformed to the scene first, so the arc is circular on
    screen) with ``angle = curvature * PI``. Positive ``curvature`` bends to the
    right of the direction of travel, negative to the left, ``0`` is straight
    (ggplot2's convention). When an arc leaves the panel it is sampled and
    clipped in scene space (the mark is then a polyline)."""

    extra_params: tuple[str, ...] = ("arrow", "lineend", "curvature", "angle", "ncp")

    def draw_panel(
        self,
        data,
        panel_params,
        coord,
        theme: Theme | None = None,
        arrow=None,
        curvature: float = 0.5,
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
        angle = float(curvature) * PI
        out, drawn = VGroup(), []
        for i, row in enumerate(data.to_dict("records")):
            color: Any = resolve_color(row.get("color"), fg)
            width = num(row.get("linewidth"), 2.0)
            alpha = num(row.get("alpha"), 1.0)
            if np.allclose(a[i], b[i]):
                continue
            arc: VMobject
            if abs(angle) < 1e-9:
                arc = Line(a[i], b[i])
            else:
                arc = ArcBetweenPoints(a[i], b[i], angle=angle)
            # outward tangents at both ends
            start_dir = arc.points[0] - arc.points[1]
            end_dir = arc.points[-1] - arc.points[-2]
            start, end = a[i], b[i]
            if box is not None:
                ts = np.linspace(0, 1, _CURVE_SAMPLES + 1)
                pts = np.array([arc.point_from_proportion(t) for t in ts])
                if not inside(pts[:, 0], pts[:, 1], box).all():
                    pieces = clip_polyline(pts[:, 0], pts[:, 1], box)
                    if not pieces:
                        continue
                    z = float(a[i][2])
                    arc = multi_path_mobject(
                        [np.column_stack([px, py, np.full(len(px), z)]) for px, py in pieces],
                        color,
                        width,
                        alpha,
                    )
                    first_in = inside(pts[0, 0], pts[0, 1], box)[()]
                    last_in = inside(pts[-1, 0], pts[-1, 1], box)[()]
                    start = start if first_in else None
                    end = end if last_in else None
            arc.set_fill(opacity=0)
            arc.set_stroke(color=color, width=width, opacity=alpha)
            line = apply_linetype(arc, row.get("linetype"))
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
    """
    return _layer(GeomSegment(), mapping, data, stat, position, inherit_aes, name, params, arrow)


@register
def geom_curve(
    mapping=None,
    data=None,
    *,
    curvature: float = 0.5,
    arrow: ArrowSpec | None = None,
    stat="identity",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params: Any,
):
    """Curved line segments from ``(x, y)`` to ``(xend, yend)``: circular arcs
    with ``angle = curvature * PI`` (``1`` = a half circle; negative values
    bend left). ``arrow=`` as for :func:`geom_segment`; ``angle``/``ncp`` are
    accepted for ggplot2 compatibility and ignored."""
    params["curvature"] = float(curvature)
    return _layer(GeomCurve(), mapping, data, stat, position, inherit_aes, name, params, arrow)
