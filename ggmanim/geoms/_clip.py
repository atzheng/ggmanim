"""Geometric clipping of marks to the panel rectangle (pure numpy).

The coordinate system's ``clip="on"`` (the default) means nothing is drawn
outside the panel. Geoms clip in *transformed* (scale) space, before calling
``coord.transform``: the panel maps transformed space linearly to the scene, so
clipping there is exact. ``rect`` is always ``(xmin, xmax, ymin, ymax)``, the
panel's ``x_range``/``y_range`` in aesthetic orientation (``coord_flip`` swaps the
axes only when drawing, so geoms never need to know about it).

* :func:`inside` -- mask of points inside the rectangle (points are dropped).
* :func:`clip_segments` -- Liang-Barsky, vectorised over segments.
* :func:`clip_polyline` -- an open path -> the list of its visible pieces.
* :func:`clip_polygon` -- Sutherland-Hodgman against the (convex) rectangle.
* :func:`clip_rects` -- axis-aligned rectangles clamped to the panel.
* :func:`panel_rect` -- the clipping rectangle for a built panel, or ``None``
  when clipping is off.

All functions treat the rectangle as closed and allow a tiny relative
tolerance, so points exactly on the border count as inside.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "Rect",
    "panel_rect",
    "inside",
    "clip_segments",
    "clip_polyline",
    "clip_polygon",
    "clip_rects",
]

Rect = tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)


def panel_rect(coord: Any, panel_params: Any) -> Rect | None:
    """``(xmin, xmax, ymin, ymax)`` of the panel, or ``None`` when ``coord.clip`` is
    ``"off"`` or there are no ``panel_params`` (nothing to clip against)."""
    if panel_params is None or getattr(coord, "clip", "on") == "off":
        return None
    (x0, x1), (y0, y1) = panel_params.x_range, panel_params.y_range
    return (float(min(x0, x1)), float(max(x0, x1)), float(min(y0, y1)), float(max(y0, y1)))


def _tol(rect: Rect) -> float:
    xmin, xmax, ymin, ymax = rect
    return 1e-9 * max(xmax - xmin, ymax - ymin, abs(xmin), abs(xmax), abs(ymin), abs(ymax), 1e-12)


def _grow(rect: Rect) -> Rect:
    t = _tol(rect)
    xmin, xmax, ymin, ymax = rect
    return (xmin - t, xmax + t, ymin - t, ymax + t)


def inside(x: Any, y: Any, rect: Rect) -> np.ndarray:
    """Boolean mask: ``(x, y)`` inside the closed rectangle (NaN -> False)."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    xmin, xmax, ymin, ymax = _grow(rect)
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def clip_segments(
    x0: Any, y0: Any, x1: Any, y1: Any, rect: Rect
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Liang-Barsky clipping of the segments ``(x0, y0) -> (x1, y1)``.

    Returns ``(t0, t1, visible)``: the visible part of segment ``i`` is the
    parameter interval ``[t0[i], t1[i]]`` (``0 <= t0 <= t1 <= 1``) where
    ``visible[i]``. Segments with a NaN end are invisible.
    """
    x0, y0, x1, y1 = (np.asarray(v, dtype=float) for v in np.broadcast_arrays(x0, y0, x1, y1))
    xmin, xmax, ymin, ymax = rect
    tol = _tol(rect)
    dx, dy = x1 - x0, y1 - y0
    t0 = np.zeros(dx.shape)
    t1 = np.ones(dx.shape)
    visible = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(x1) & np.isfinite(y1)
    with np.errstate(divide="ignore", invalid="ignore"):
        for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
            visible &= ~((p == 0) & (q < -tol))  # parallel to and outside this edge
            r = q / p
            t0 = np.where(p < 0, np.maximum(t0, r), t0)
            t1 = np.where(p > 0, np.minimum(t1, r), t1)
    visible &= t0 <= t1 + 1e-12
    t1 = np.maximum(t0, t1)
    return t0, t1, visible


def clip_polyline(x: Any, y: Any, rect: Rect) -> list[tuple[np.ndarray, np.ndarray]]:
    """Clip the open path through ``(x, y)`` to ``rect``.

    Returns the visible pieces as ``[(xs, ys), ...]`` in path order; a piece
    starts/ends on the border where the path crosses it. A path entirely inside
    is returned unchanged (same vertices); one entirely outside gives ``[]``.
    A single point is kept iff it is inside.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) == 0:
        return []
    ins = inside(x, y, rect)
    if ins.all():
        return [(x.copy(), y.copy())]
    if len(x) == 1:
        return []
    x0, y0, x1, y1 = x[:-1], y[:-1], x[1:], y[1:]
    t0, t1, vis = clip_segments(x0, y0, x1, y1, rect)
    cx0, cy0 = x0 + t0 * (x1 - x0), y0 + t0 * (y1 - y0)
    cx1, cy1 = x0 + t1 * (x1 - x0), y0 + t1 * (y1 - y0)
    # Use the exact original vertices where the segment was not cut.
    cx0, cy0 = np.where(t0 <= 0, x0, cx0), np.where(t0 <= 0, y0, cy0)
    cx1, cy1 = np.where(t1 >= 1, x1, cx1), np.where(t1 >= 1, y1, cy1)

    pieces: list[tuple[list[float], list[float]]] = []
    joined = False  # the previous segment reached its end vertex uncut
    for i in range(len(x0)):
        if not vis[i]:
            joined = False
            continue
        if not (joined and t0[i] <= 0):
            pieces.append(([float(cx0[i])], [float(cy0[i])]))
        px, py = pieces[-1]
        if (cx1[i], cy1[i]) != (px[-1], py[-1]):  # skip zero-length (border-touching) bits
            px.append(float(cx1[i]))
            py.append(float(cy1[i]))
        joined = bool(t1[i] >= 1)
    # a piece that only touches the border at one point is invisible
    return [(np.asarray(px), np.asarray(py)) for px, py in pieces if len(px) >= 2]


def clip_polygon(x: Any, y: Any, rect: Rect) -> tuple[np.ndarray, np.ndarray]:
    """Sutherland-Hodgman clipping of the closed polygon ``(x, y)`` to ``rect``.

    Returns the clipped polygon's vertices (empty arrays when nothing is
    visible). A polygon entirely inside is returned unchanged. Non-finite
    vertices are dropped first.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return np.zeros(0), np.zeros(0)
    if inside(x, y, rect).all():
        return x.copy(), y.copy()
    xmin, xmax, ymin, ymax = rect
    pts = np.column_stack([x, y])
    # (axis, bound, keep_greater): keep points with pts[:, axis] >= / <= bound
    for axis, bound, ge in ((0, xmin, True), (0, xmax, False), (1, ymin, True), (1, ymax, False)):
        if len(pts) == 0:
            break
        d = pts[:, axis] - bound
        ins = d >= 0 if ge else d <= 0
        prev = np.roll(pts, 1, axis=0)
        dprev = np.roll(d, 1)
        insprev = np.roll(ins, 1)
        out = []
        for k in range(len(pts)):
            if ins[k] != insprev[k]:  # edge prev -> k crosses the boundary
                t = dprev[k] / (dprev[k] - d[k])
                p = prev[k] + t * (pts[k] - prev[k])
                p[axis] = bound
                out.append(p)
            if ins[k]:
                out.append(pts[k])
        pts = np.asarray(out, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros(0), np.zeros(0)
    return pts[:, 0].copy(), pts[:, 1].copy()


def clip_rects(
    xmin: Any, xmax: Any, ymin: Any, ymax: Any, rect: Rect
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clamp axis-aligned rectangles to ``rect``.

    Returns ``(xmin, xmax, ymin, ymax, visible)``; ``visible`` is False for
    rectangles that do not overlap the panel at all (touching the border counts as
    overlapping, so zero-height bars on the border are kept).
    """
    a, b, c, d = (np.asarray(v, dtype=float) for v in np.broadcast_arrays(xmin, xmax, ymin, ymax))
    lo_x, hi_x = np.minimum(a, b), np.maximum(a, b)
    lo_y, hi_y = np.minimum(c, d), np.maximum(c, d)
    gx0, gx1, gy0, gy1 = _grow(rect)
    visible = (hi_x >= gx0) & (lo_x <= gx1) & (hi_y >= gy0) & (lo_y <= gy1)
    rx0, rx1, ry0, ry1 = rect
    return (
        np.clip(a, rx0, rx1),
        np.clip(b, rx0, rx1),
        np.clip(c, ry0, ry1),
        np.clip(d, ry0, ry1),
        visible,
    )
