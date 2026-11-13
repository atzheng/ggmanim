"""Clip point markers to the panel (scene space).

:mod:`ggmanim.geoms._clip` clips in transformed (data) space, which is exact for
marks whose geometry *is* data. A marker's geometry is in scene units around its
centre, so it is clipped after drawing, against the panel's rectangle in the
scene:

* :func:`scene_box` -- the panel rectangle ``rect`` mapped to scene coordinates.
* :func:`near_box` -- mask of marker centres close enough to the panel to be
  (partly) visible.
* :func:`clip_marker` -- a marker clipped to the box. Markers entirely inside are
  returned as is (the fast path, so interior points are untouched); markers
  crossing the border are rebuilt from their outlines (bezier curves sampled to
  polygons): the fill is clipped as a polygon (Sutherland-Hodgman) and the
  outline as an open path, so the border itself is never stroked.

As for every other clipped geom, the *geometry* is clipped exactly; a stroke that
ends at (or runs along) the border may bleed over it by half its line width.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from manim import VGroup, VMobject

from ._clip import Rect, clip_polygon, clip_polyline

__all__ = ["scene_box", "near_box", "marker_extent", "clip_marker"]

# Cairo draws ``stroke_width`` as ``stroke_width * 0.01`` scene units.
_STROKE_UNIT = 0.01
# Samples per curved bezier segment (a circle has 8 segments -> 64 vertices).
_SAMPLES = 8


def scene_box(coord: Any, rect: Rect) -> Rect:
    """``rect`` (``(xmin, xmax, ymin, ymax)`` in transformed space) as the same
    tuple in scene coordinates (works for ``coord_flip``)."""
    from .base import transform_points

    xmin, xmax, ymin, ymax = rect
    c = transform_points(coord, [xmin, xmax], [ymin, ymax])
    return (
        float(c[:, 0].min()),
        float(c[:, 0].max()),
        float(c[:, 1].min()),
        float(c[:, 1].max()),
    )


def marker_extent(radius: Any) -> np.ndarray:
    """Upper bound on how far a marker of ``radius`` (incl. its outline) reaches
    from its centre: the 5-point star (outer radius ~1.68 r) is the widest shape."""
    r = np.abs(np.nan_to_num(np.asarray(radius, dtype=float), nan=0.0))
    return 1.75 * r + _STROKE_UNIT * 0.5 * 1.2 * np.clip(r * 35.0, 1.0, 8.0) + 1e-9


def near_box(pts: Any, box: Rect, margin: Any) -> np.ndarray:
    """Mask of scene points within ``margin`` (scalar or per point) of ``box``."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    m = np.broadcast_to(np.asarray(margin, dtype=float), (len(pts),))
    xmin, xmax, ymin, ymax = box
    x, y = pts[:, 0], pts[:, 1]
    return (x >= xmin - m) & (x <= xmax + m) & (y >= ymin - m) & (y <= ymax + m)


def _bernstein(n: int) -> np.ndarray:
    """``(_SAMPLES, n)`` weights evaluating a degree ``n - 1`` bezier at
    ``t = 0, 1/_SAMPLES, ...`` (``t = 1`` excluded)."""
    t = np.linspace(0.0, 1.0, _SAMPLES, endpoint=False)[:, None]
    k = np.arange(n)[None, :]
    binom = np.array([math.comb(n - 1, i) for i in range(n)], dtype=float)[None, :]
    return np.asarray(binom * t**k * (1 - t) ** (n - 1 - k), dtype=float)


_BERNSTEIN = {n: _bernstein(n) for n in (2, 3, 4)}


def _sample_subpath(sub: np.ndarray, n_per_curve: int) -> np.ndarray:
    """Polygon vertices (``(k, 2)``) along the bezier subpath ``sub``; straight
    segments contribute only their start point."""
    out = []
    for i in range(0, len(sub) - n_per_curve + 1, n_per_curve):
        ctrl = sub[i : i + n_per_curve, :2]
        a, b = ctrl[0], ctrl[-1]
        chord = b - a
        length = float(np.hypot(*chord))
        if length > 0:
            d = ctrl - a
            dev = np.abs(chord[0] * d[:, 1] - chord[1] * d[:, 0]) / length
        else:
            dev = np.hypot(*(ctrl - a).T)
        if float(dev.max()) <= 1e-9 * max(length, 1.0):
            out.append(a[None, :])
        else:
            out.append(_BERNSTEIN[len(ctrl)] @ ctrl)
    if len(sub):
        out.append(sub[-1:, :2])
    return np.concatenate(out) if out else np.zeros((0, 2))


def _snap(xy: np.ndarray, box: Rect, tol: float) -> np.ndarray:
    """Move vertices within ``tol`` of a box edge onto it, so outlines running
    along the border (e.g. a ``plus`` arm, up to rounding) are not mis-clipped."""
    xy = xy.copy()
    for axis, bounds in ((0, box[:2]), (1, box[2:])):
        for b in bounds:
            near = np.abs(xy[:, axis] - b) <= tol
            xy[near, axis] = b
    return xy


def _corners_mobject(pieces: list[np.ndarray], z: float, closed: bool) -> VMobject:
    v = VMobject()
    for xy in pieces:
        p = np.column_stack([xy, np.full(len(xy), z)])
        if closed:
            p = np.vstack([p, p[:1]])
        v.start_new_path(p[0])
        v.add_points_as_corners(p[1:])
    return v


def clip_marker(marker: VMobject, box: Rect) -> VMobject | None:
    """``marker`` clipped to the scene rectangle ``box``.

    Returns ``marker`` itself when it lies entirely inside, ``None`` when nothing
    of it is visible, and otherwise a ``VGroup`` of the visible fill polygon(s)
    and stroke pieces, styled like the corresponding parts of ``marker``.
    """
    members = [m for m in marker.family_members_with_points()]
    if not members:
        return marker
    pts = np.concatenate([m.points for m in members])
    xmin, xmax, ymin, ymax = box
    lo, hi = pts[:, :2].min(axis=0), pts[:, :2].max(axis=0)
    tol = 1e-9 * max(xmax - xmin, ymax - ymin, 1.0)
    if lo[0] >= xmin - tol and hi[0] <= xmax + tol and lo[1] >= ymin - tol and hi[1] <= ymax + tol:
        return marker  # fast path: entirely inside

    parts = VGroup()
    for m in members:
        z = float(m.points[0, 2])
        n = int(getattr(m, "n_points_per_cubic_curve", 4))
        subs = [_sample_subpath(np.asarray(s), n) for s in m.get_subpaths()]
        subs = [_snap(s, box, tol) for s in subs if len(s) >= 2]
        fill_pieces: list[np.ndarray] = []
        stroke_pieces: list[np.ndarray] = []
        for s in subs:
            closed = bool(np.allclose(s[0], s[-1]))
            if m.get_fill_opacity() > 0 and closed:
                fx, fy = clip_polygon(s[:-1, 0], s[:-1, 1], box)
                if len(fx) >= 3:
                    fill_pieces.append(np.column_stack([fx, fy]))
            if m.get_stroke_width() > 0 and m.get_stroke_opacity() > 0:
                pieces = [np.column_stack(p) for p in clip_polyline(s[:, 0], s[:, 1], box)]
                if (
                    closed
                    and len(pieces) > 1
                    and np.array_equal(pieces[0][0], s[0])
                    and np.array_equal(pieces[-1][-1], s[-1])
                ):  # the outline's start vertex is visible: rejoin the pieces there
                    pieces = [np.vstack([pieces[-1], pieces[0][1:]]), *pieces[1:-1]]
                stroke_pieces.extend(pieces)
        if fill_pieces:
            f = _corners_mobject(fill_pieces, z, closed=True)
            f.set_fill(m.get_fill_color(), opacity=m.get_fill_opacity())
            f.set_stroke(width=0, opacity=0)
            parts.add(f)
        if stroke_pieces:
            s_mob = _corners_mobject(stroke_pieces, z, closed=False)
            s_mob.set_fill(opacity=0)
            s_mob.set_stroke(
                m.get_stroke_color(), width=m.get_stroke_width(), opacity=m.get_stroke_opacity()
            )
            parts.add(s_mob)
    if not len(parts):
        return None
    if hasattr(marker, "marker_shape"):
        parts.marker_shape = marker.marker_shape  # type: ignore[attr-defined]
    return parts
