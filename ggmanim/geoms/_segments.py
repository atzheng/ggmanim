"""Per-segment drawing for path geoms whose aesthetics vary along a group.

ggplot2's ``geom_path``/``geom_line`` draws a group as one polyline when its
``colour``, ``linewidth`` and ``alpha`` are constant, and otherwise as one
segment per consecutive pair of vertices, each styled by its *starting* vertex
(so the last vertex's aesthetics are never used). :func:`segmented_path` does
the latter; :func:`varies_within` decides which case applies.

Why segments rather than manim's multi-colour strokes: ``set_stroke(color=[...])``
/ ``set_color_by_gradient`` are rendered by the Cairo camera as one *linear*
gradient between two points of the mobject's bounding box, not along the path,
so a spiral or any non-monotone line would get the wrong colours. Separate
segment ``VMobject``\\ s colour exactly like ggplot. Consecutive segments that
share a style (and join uncut) are merged into one polyline, so a group whose
colour changes only a few times stays cheap. Where an opaque run meets the
next one, it extends one stroke width under it so the joint shows no
anti-aliasing seam (the next run, drawn later, covers the overlap).

Clipping is per segment (Liang-Barsky, :func:`._clip.clip_segments`), in
transformed space like the rest of the geoms. Dashing (``linetype``) is measured
by arc length along the *whole* unclipped group path in scene units, so the
pattern continues across segment and style boundaries and is unaffected by
where the panel cuts the path (ggplot2/grid restarts it on every segment). The
pattern unit comes from the group's first ``linewidth``. As in ggplot2, a
``linetype`` that varies within a group is an error.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from manim import ManimColor, VGroup, VMobject

from ..scales.shape import parse_linetype
from ._clip import clip_segments, panel_rect
from ._linetype import DASH_UNIT
from .base import _scalar, num, resolve_color, transform_points

__all__ = ["VARYING_AES", "varies_within", "dash_polyline_phase", "segmented_path"]

#: Aesthetics that switch a group from one polyline to per-segment drawing.
VARYING_AES = ("color", "linewidth", "alpha")


def _key(v: Any) -> Any:
    v = _scalar(v)
    return v.to_hex() if isinstance(v, ManimColor) else v


def _n_distinct(col: pd.Series) -> int:
    return len({repr(_key(v)) for v in col})


def varies_within(gdf: pd.DataFrame, cols: tuple[str, ...] = VARYING_AES) -> bool:
    """Whether any of ``cols`` takes more than one value in the group ``gdf``.

    Raises ``ValueError`` (like ggplot2) when ``linetype`` varies within it.
    """
    if len(gdf) < 2:
        return False
    if "linetype" in gdf.columns and _n_distinct(gdf["linetype"]) > 1:
        raise ValueError(
            "geom_path/geom_line: linetype must be constant within a group "
            f"(group {_key(gdf['group'].iloc[0]) if 'group' in gdf else '?'} has several)"
        )
    return any(c in gdf.columns and _n_distinct(gdf[c]) > 1 for c in cols)


def dash_polyline_phase(
    points: np.ndarray, pattern: tuple[float, ...], phase: float = 0.0
) -> list[np.ndarray]:
    """Like :func:`._linetype.dash_polyline`, but the polyline starts ``phase``
    scene units into the (on/off, even-length) ``pattern``."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2 or not pattern:
        return []
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    total = s[-1]
    if total <= 0:
        return []
    m = len(pattern)
    u = float(phase) % float(sum(pattern))
    k = 0
    while k < m - 1 and u >= pattern[k]:
        u -= pattern[k]
        k += 1

    def at(a: float) -> np.ndarray:
        return np.array([np.interp(a, s, points[:, j]) for j in range(points.shape[1])])

    dashes = []
    t, i = -u, k
    while t < total:
        length = pattern[i % m]
        if i % 2 == 0:
            a, b = max(t, 0.0), min(t + length, total)
            if b > a:
                dashes.append(np.vstack([at(a), points[(s > a) & (s < b)], at(b)]))
        t += length
        i += 1
    return dashes


def _stroke(points: np.ndarray, style: tuple[Any, float, float]) -> VMobject:
    color, width, alpha = style
    m = VMobject()
    m.set_points_as_corners(points)
    m.set_fill(opacity=0)
    m.set_stroke(color=color, width=width, opacity=alpha)
    return m


def segmented_path(coord: Any, panel_params: Any, gdf: pd.DataFrame, fg: Any) -> VGroup:
    """The group ``gdf`` (finite ``x``/``y`` in transformed space, in drawing
    order, with ``color``/``linewidth``/``alpha``/``linetype`` columns) as one
    stroke per segment ``i -> i+1`` styled by row ``i``, clipped to the panel.

    Returns a ``VGroup`` of polylines (runs of joined same-style segments) or,
    for a dashed ``linetype``, of dashes (``.linetype`` set, as
    :func:`._linetype.apply_linetype` does).
    """
    x = gdf["x"].to_numpy(float)
    y = gdf["y"].to_numpy(float)
    out: Any = VGroup()
    if len(x) < 2:
        return out
    styles = [
        (resolve_color(_scalar(c), fg), num(w, 3.0), num(a, 1.0))
        for c, w, a in zip(gdf["color"], gdf["linewidth"], gdf["alpha"])
    ]
    keys = [(c.to_hex() if c is not None else None, w, a) for c, w, a in styles]
    linetype = _scalar(gdf["linetype"].iloc[0]) if "linetype" in gdf else None
    pattern = parse_linetype(linetype)

    x0, y0, x1, y1 = x[:-1], y[:-1], x[1:], y[1:]
    rect = panel_rect(coord, panel_params)
    if rect is None:
        t0, t1 = np.zeros(len(x0)), np.ones(len(x0))
    else:
        t0, t1, vis = clip_segments(x0, y0, x1, y1, rect)
        t0, t1 = np.where(vis, t0, np.nan), np.where(vis, t1, np.nan)
    # exact original vertices where a segment is not cut
    cx0 = np.where(t0 <= 0, x0, x0 + t0 * (x1 - x0))
    cy0 = np.where(t0 <= 0, y0, y0 + t0 * (y1 - y0))
    cx1 = np.where(t1 >= 1, x1, x0 + t1 * (x1 - x0))
    cy1 = np.where(t1 >= 1, y1, y0 + t1 * (y1 - y0))
    full = transform_points(coord, x, y)
    seg_len = np.linalg.norm(np.diff(full, axis=0), axis=1)
    offset = np.concatenate([[0.0], np.cumsum(seg_len)])[:-1] + np.nan_to_num(t0) * seg_len
    a_pts = transform_points(coord, cx0, cy0)
    b_pts = transform_points(coord, cx1, cy1)

    # runs: [first segment index, points, style index]
    runs: list[tuple[int, list[np.ndarray], int]] = []
    prev = -2
    for i in range(len(x0)):
        if not np.isfinite(t0[i]):
            continue
        if np.allclose(a_pts[i], b_pts[i]):  # zero length: invisible, but keeps a run joined
            if prev == i - 1 and t0[i] <= 0 and t1[i] >= 1:
                prev = i
            continue
        joined = prev == i - 1 and t1[prev] >= 1 and t0[i] <= 0
        if runs and joined and keys[runs[-1][2]] == keys[i]:
            runs[-1][1].append(b_pts[i])
        else:
            if runs and joined and styles[runs[-1][2]][2] >= 1 and styles[i][2] >= 1:
                # Overlap the previous (opaque) run a stroke width into this
                # segment; this run is drawn on top, so the joint shows no
                # anti-aliasing seam. (Skipped for translucent strokes, where
                # the overlap would show as a darker patch.)
                d = b_pts[i] - a_pts[i]
                n = float(np.linalg.norm(d))
                ext = min(n, DASH_UNIT * styles[runs[-1][2]][1])
                runs[-1][1].append(a_pts[i] + d * (ext / n))
            runs.append((i, [a_pts[i], b_pts[i]], i))
        prev = i

    if pattern is not None and len(pattern) > 0:
        unit = DASH_UNIT * max(styles[0][1], 0.5)
        scaled = tuple(p * unit for p in pattern)
        for first_i, pts, si in runs:
            for dash in dash_polyline_phase(np.asarray(pts), scaled, offset[first_i]):
                out.add(_stroke(dash, styles[si]))
        out.linetype = linetype
        return out
    for _, pts, si in runs:
        m = _stroke(np.asarray(pts), styles[si])
        if pattern is not None:  # "blank"
            m.set_stroke(opacity=0)
        out.add(m)
    return out
