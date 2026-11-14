"""Point markers for the ``shape`` aesthetic.

:func:`make_marker` draws one marker. Shape names are parsed by
:func:`ggmanim.scales.shape.parse_shape`; the manim classes used are

=========== =================================================
circle      ``Dot``
triangle    ``Triangle``
square      ``Square``
diamond     ``RegularPolygon`` (4 vertices, pointing up)
star        ``Star`` (5 points)
cross / plus ``Cross`` (two stroked lines; ``plus`` is rotated 45 degrees)
=========== =================================================

Filled shapes are normalised to the same *area* as a circle of radius ``size``,
so a triangle and a circle with equal ``size`` look equally heavy. The
stroke-only ``cross``/``plus`` span about the circle's diameter.

Styles: ``solid`` shapes are filled with ``fill`` (falling back to ``color``)
and have no outline; ``*_open`` shapes are an outline in ``color``;
``*_filled`` shapes are outlined in ``color`` and filled with ``fill`` (unfilled
if ``fill`` is ``None``); ``cross``/``plus`` are stroked in ``color``.

Each marker is a copy of a cached unit-size prototype per shape (scaled and
shifted), which is several times faster than constructing manim polygons row by
row; the copy is :func:`ggmanim._text_cache.fast_copy`, a deep copy without the
generic ``copy.deepcopy`` machinery.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from manim import ORIGIN, PI, Cross, Dot, RegularPolygon, Square, Star, Triangle, VMobject

from .._text_cache import fast_copy
from ..scales.shape import parse_shape

__all__ = ["make_marker", "outline_width"]

_PROTOTYPES: dict[str, VMobject] = {}


def _polygon_area(vertices: np.ndarray) -> float:
    x, y = vertices[:, 0], vertices[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _prototype(base: str) -> VMobject:
    """A unit marker of ``base`` centred at the origin, with area ``pi`` (the area
    of a unit circle) for fillable shapes."""
    proto = _PROTOTYPES.get(base)
    if proto is not None:
        return proto
    m: VMobject
    if base == "circle":
        m = Dot(ORIGIN, radius=1.0)
    elif base == "square":
        m = Square(side_length=math.sqrt(math.pi))
    elif base == "triangle":
        m = Triangle()  # circumradius 1, centroid at the origin
    elif base == "diamond":
        m = RegularPolygon(4)  # vertices on the axes
    elif base == "star":
        m = Star(5, outer_radius=1.0)
    elif base == "cross":
        m = Cross(stroke_width=1).scale(0.85)  # half-diagonal ~1.2
    elif base == "plus":
        m = Cross(stroke_width=1).scale(0.85).rotate(PI / 4)
    else:  # pragma: no cover - parse_shape guarantees a known base
        raise ValueError(base)
    if base in ("triangle", "diamond", "star"):
        m.scale(math.sqrt(math.pi / _polygon_area(m.get_vertices())), about_point=ORIGIN)
    _PROTOTYPES[base] = m
    return m


def outline_width(radius: float) -> float:
    """Default outline stroke width (manim units) for a marker of ``radius``."""
    return float(min(max(radius * 35.0, 1.0), 8.0))


def make_marker(
    shape: Any,
    point: Any,
    radius: float,
    color: Any,
    fill: Any = None,
    alpha: float = 1.0,
    stroke_width: float | None = None,
) -> VMobject:
    """One marker for ``shape`` centred at ``point``.

    ``radius`` is the radius (scene units) of the circle with the same area;
    ``color`` is the outline / solid colour, ``fill`` the fill (``None``: not set,
    solid shapes then use ``color``); ``alpha`` applies to fill and stroke;
    ``stroke_width`` overrides :func:`outline_width` for outlined shapes.
    Raises ``ValueError`` for unknown shapes.
    """
    base, style = parse_shape(shape)
    radius = max(float(radius), 0.0)
    m = fast_copy(_prototype(base))
    assert isinstance(m, VMobject)
    # ``m.scale(k, about_point=ORIGIN).shift(point)`` on the points directly (the
    # same arithmetic, without manim's per-call family walks)
    k = radius if radius > 0 else 1e-6
    offset = np.asarray(point, dtype=float)
    for mob in m.family_members_with_points() if m.submobjects else (m,):
        pts = k * mob.points
        pts += ORIGIN
        pts += offset
        mob.points = pts
    if isinstance(m, Dot):
        m.radius = radius
    sw = outline_width(radius) if stroke_width is None else float(stroke_width)
    if style == "solid":
        m.set_fill(fill if fill is not None else color, opacity=alpha)
        m.set_stroke(width=0, opacity=0)
    elif style == "open":
        m.set_fill(color, opacity=0)
        m.set_stroke(color, width=sw, opacity=alpha)
    elif style == "filled":
        if fill is None:
            m.set_fill(color, opacity=0)
        else:
            m.set_fill(fill, opacity=alpha)
        m.set_stroke(color, width=sw, opacity=alpha)
    else:  # stroke-only cross / plus
        m.set_fill(color, opacity=0)
        m.set_stroke(color, width=sw * 1.2, opacity=alpha)
    m.marker_shape = base  # type: ignore[attr-defined]
    return m
