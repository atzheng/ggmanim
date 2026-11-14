"""Shared drawing helpers for filled geoms (rect, tile, raster, ribbon, area, polygon).

Everything here works on *scene* points (``coord.transform`` output) except
:func:`draw_rects`, which clips transformed-space rectangles to the panel first.

* :func:`ring_points` / :func:`rings_mobject` -- closed polylines as straight
  cubic Bezier subpaths of one ``VMobject`` (built straight from a numpy array,
  much faster than ``Polygon(*corners)`` for thousands of tiles).
* :func:`orient_rings` -- give every ring the orientation that makes manim's
  (Cairo's) *nonzero winding* fill behave like the *even-odd* rule ggplot2 uses
  for polygon holes: rings nested an even number of times run counter-clockwise,
  odd ones clockwise. Manim exposes no fill-rule setting, so this is how
  ``geom_polygon(subgroup=...)`` gets its holes.
* :func:`style_filled` -- fill/alpha/outline/linewidth/linetype. Like ggplot2's
  ``GeomRect``/``GeomPolygon``/``GeomRibbon``, ``alpha`` applies to the fill only;
  ``color=None`` (the default) means no outline. A non-solid ``linetype`` turns
  the result into ``VGroup(fill, dashes)``.
* :func:`rect_corners` / :func:`draw_rects` -- axis-aligned rectangles (one
  mobject per visible row), clipped with :func:`._clip.clip_rects`.
* :func:`filled_key` -- the legend glyph (``draw_key_polygon``): a filled square.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from manim import Square, VGroup, VMobject

from ..theme import Theme
from ._clip import clip_rects, panel_rect
from ._linetype import apply_linetype
from .base import KEY_SIZE, munch_rect_corners, num, paint_fill, resolve_color

__all__ = [
    "ring_points",
    "rings_mobject",
    "orient_rings",
    "signed_area",
    "style_filled",
    "seal_seams",
    "rect_corners",
    "clipped_rects",
    "draw_rects",
    "filled_key",
]


def ring_points(ring: np.ndarray) -> np.ndarray:
    """Bezier control points (``4 * k`` rows) of the closed polyline through the
    ``k`` points of ``ring`` (``(k, 3)``), each edge a straight cubic curve."""
    ring = np.asarray(ring, dtype=float).reshape(-1, 3)
    a = ring
    b = np.roll(ring, -1, axis=0)
    out = np.empty((len(ring), 4, 3))
    out[:, 0] = a
    out[:, 1] = a + (b - a) / 3.0
    out[:, 2] = a + 2.0 * (b - a) / 3.0
    out[:, 3] = b
    return out.reshape(-1, 3)


def rings_mobject(rings: Sequence[np.ndarray]) -> VMobject:
    """One ``VMobject`` whose subpaths are the closed ``rings`` (each ``(k, 3)``,
    ``k >= 3``; shorter rings are skipped)."""
    m = VMobject()
    parts = [ring_points(r) for r in rings if len(r) >= 3]
    if parts:
        m.set_points(np.concatenate(parts, axis=0))
    return m


def signed_area(ring: np.ndarray) -> float:
    """Shoelace area of ``ring`` in the xy plane (positive = counter-clockwise)."""
    ring = np.asarray(ring, dtype=float)
    if len(ring) < 3:
        return 0.0
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _point_in_ring(px: float, py: float, ring: np.ndarray) -> bool:
    """Even-odd ray casting: is ``(px, py)`` inside ``ring``?"""
    x, y = ring[:, 0], ring[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    crosses = (y > py) != (y2 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x + (py - y) * (x2 - x) / (y2 - y)
    return bool(np.count_nonzero(crosses & (px < xi)) % 2)


def orient_rings(rings: Sequence[np.ndarray]) -> list[np.ndarray]:
    """``rings`` re-oriented so the nonzero fill rule gives the even-odd result:
    a ring inside an even number of the others (outer boundaries, islands in
    holes) runs counter-clockwise, one inside an odd number (holes) clockwise.
    Nesting is decided by each ring's first vertex (rings are assumed not to
    cross each other, as for ggplot2's ``subgroup`` holes)."""
    rings = [np.asarray(r, dtype=float) for r in rings]
    if len(rings) <= 1:
        return [r if signed_area(r) >= 0 else r[::-1] for r in rings]
    out = []
    for i, r in enumerate(rings):
        if len(r) == 0:
            out.append(r)
            continue
        depth = sum(
            _point_in_ring(r[0, 0], r[0, 1], other)
            for j, other in enumerate(rings)
            if j != i and len(other) >= 3
        )
        ccw = signed_area(r) >= 0
        out.append(r if ccw == (depth % 2 == 0) else r[::-1])
    return out


def style_filled(
    mob: VMobject, row: Mapping[str, Any], theme: Theme, default_linewidth: float = 1.0
) -> VMobject:
    """Fill ``mob`` with ``row["fill"]`` at opacity ``alpha``; stroke it with
    ``row["color"]`` (no outline when ``None``), ``linewidth`` and ``linetype``.

    Returns ``mob`` itself, or ``VGroup(mob, dashes)`` for a dashed outline."""
    alpha = num(row.get("alpha"), 1.0)
    paint_fill(mob, resolve_color(row.get("fill"), theme.get_foreground()), alpha)
    outline = resolve_color(row.get("color"))
    if outline is None:
        mob.set_stroke(width=0, opacity=0)
        return mob
    mob.set_stroke(color=outline, width=num(row.get("linewidth"), default_linewidth), opacity=1.0)
    linetype = row.get("linetype")
    if linetype is None or linetype == "solid":
        return mob
    dashes = apply_linetype(mob.copy().set_fill(opacity=0), linetype)
    if not isinstance(dashes, VGroup):  # the copy came back as is: solid or "blank"
        if dashes.get_stroke_opacity() == 0:
            mob.set_stroke(width=0, opacity=0)
        return mob
    mob.set_stroke(width=0, opacity=0)
    return VGroup(mob, dashes)


#: Stroke width (manim units) of the fill-coloured hairline :func:`seal_seams` adds.
SEAM_WIDTH = 1.0


def seal_seams(mob: VMobject, width: float = SEAM_WIDTH) -> VMobject:
    """Give an opaque, outline-less ``mob`` a hairline stroke in its fill colour.

    Abutting tiles are anti-aliased separately, so the background shows through
    as faint lines along their shared edges; the hairline overlaps neighbours by
    a fraction of a pixel and hides them. Translucent fills (where the overlap
    would show) and mobjects with an outline are left alone."""
    if mob.get_stroke_width() == 0 and mob.get_fill_opacity() >= 1:
        mob.set_stroke(color=mob.get_fill_color(), width=width, opacity=1.0)
    return mob


def rect_corners(coord: Any, xmin, xmax, ymin, ymax) -> np.ndarray:
    """``(n, 4, 3)`` scene corners of the transformed-space rectangles, in the
    order ``(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)``; one
    vectorised ``coord.transform`` call."""
    xmin, xmax, ymin, ymax = (np.asarray(v, dtype=float).ravel() for v in (xmin, xmax, ymin, ymax))
    cx = np.stack([xmin, xmax, xmax, xmin], axis=1)
    cy = np.stack([ymin, ymin, ymax, ymax], axis=1)
    return munch_rect_corners(coord, cx, cy)


def clipped_rects(
    data: pd.DataFrame, coord: Any, panel_params: Any
) -> tuple[pd.DataFrame, np.ndarray]:
    """The rows of ``data`` whose ``xmin``/``xmax``/``ymin``/``ymax`` rectangle is
    visible and their ``(n, 4, 3)`` scene corners (clamped to the panel when
    clipping is on)."""
    xmin, xmax, ymin, ymax = (data[c].to_numpy(float) for c in ("xmin", "xmax", "ymin", "ymax"))
    rect = panel_rect(coord, panel_params)
    if rect is not None:
        xmin, xmax, ymin, ymax, vis = clip_rects(xmin, xmax, ymin, ymax, rect)
        xmin, xmax, ymin, ymax, data = xmin[vis], xmax[vis], ymin[vis], ymax[vis], data[vis]
    return data, rect_corners(coord, xmin, xmax, ymin, ymax)


def draw_rects(
    data: pd.DataFrame, panel_params: Any, coord: Any, theme: Theme, seal: bool = False
) -> tuple[VGroup, pd.DataFrame]:
    """One styled mobject per visible rectangle row of ``data`` (which must have
    ``xmin``/``xmax``/``ymin``/``ymax`` and the aesthetic columns). Returns the
    ``VGroup`` and the drawn rows (for :func:`~.base.attach_marks`). ``seal``
    applies :func:`seal_seams` to each (for tiles that abut)."""
    data, corners = clipped_rects(data, coord, panel_params)
    out = VGroup()
    rows = data.to_dict("records")
    for c, row in zip(corners, rows):
        m = VMobject()
        m.set_points(ring_points(c))
        m = style_filled(m, row, theme)
        out.add(seal_seams(m) if seal and not isinstance(m, VGroup) else m)
    return out, data


def filled_key(row: Mapping[str, Any], theme: Theme, default_linewidth: float = 1.0) -> VMobject:
    """Legend glyph for filled geoms: a :data:`~.base.KEY_SIZE` square styled
    like the geom (fill, alpha, outline)."""
    return style_filled(Square(side_length=KEY_SIZE), row, theme, default_linewidth)
