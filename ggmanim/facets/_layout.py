"""Panel-grid geometry shared by every facet: which panels draw axes, how big
each panel is, and where it goes.

All functions take the facet's *layout table* (see :mod:`ggmanim.facets.base`):
one row per panel with ``PANEL`` (1..n), ``ROW``/``COL`` (1-based grid cell),
``SCALE_X``/``SCALE_Y`` (which x/y scale the panel uses). Facets only compute
that table; the pipeline (``build.build_panels``) calls these helpers:

1. :func:`axis_visibility` -- per panel, draw the horizontal / vertical axis
   (tick marks + tick labels)?
2. :func:`track_sizes` -- column widths and row heights of the plotting areas,
   given each panel's decoration extents (tick labels, strips) measured on a
   first build, so the whole grid fits ``theme.panel_size``.
3. :func:`place_units` -- move each panel (with its strips) into its cell.

Scene geometry: the *plotting area* of a panel is its ``axes`` bounding box;
*extents* ``(top, right, bottom, left)`` are how far the rest of the panel unit
(tick labels, strips ...) sticks out of it on each side.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from manim import DL, UR, Mobject

__all__: list[str] = []

Extents = tuple[float, float, float, float]  # top, right, bottom, left


def grid_shape(layout: pd.DataFrame) -> tuple[int, int]:
    """``(nrow, ncol)`` of the layout table."""
    return int(layout["ROW"].max()), int(layout["COL"].max())


def axis_visibility(
    layout: pd.DataFrame,
    h_col: str = "SCALE_X",
    v_col: str = "SCALE_Y",
    h_side: str = "bottom",
    v_side: str = "left",
) -> dict[int, tuple[bool, bool]]:
    """``{PANEL: (draw horizontal axis, draw vertical axis)}``.

    A panel draws its horizontal axis unless the neighbouring panel on the axis
    side (below for ``h_side="bottom"``, above for ``"top"``) exists and uses the
    same horizontal scale (``h_col``); likewise the vertical axis with the
    panel to its left/right. So fixed scales give axes on the outer panels only
    (and on panels above a gap in an incomplete last row), free scales on every
    panel, and per-column / per-row scales (``facet_grid``) once per column/row.
    """
    cells = {(int(r.ROW), int(r.COL)): r for r in layout.itertuples(index=False)}
    dr = 1 if h_side == "bottom" else -1
    dc = -1 if v_side == "left" else 1
    out = {}
    for (row, col), r in cells.items():
        below = cells.get((row + dr, col))
        beside = cells.get((row, col + dc))
        h = below is None or getattr(below, h_col) != getattr(r, h_col)
        v = beside is None or getattr(beside, v_col) != getattr(r, v_col)
        out[int(r.PANEL)] = (bool(h), bool(v))
    return out


def area_box(panel: Mobject) -> tuple[np.ndarray, np.ndarray]:
    """``(lower-left, upper-right)`` of a panel's plotting area (its ``axes``)."""
    area = getattr(panel, "axes", panel)
    return area.get_corner(DL), area.get_corner(UR)


def extents(unit: Mobject, area: Mobject) -> Extents:
    """How far ``unit`` sticks out of ``area`` on each side (>= 0)."""
    (ulx, uly, _), (uhx, uhy, _) = unit.get_corner(DL), unit.get_corner(UR)
    (alx, aly, _), (ahx, ahy, _) = area.get_corner(DL), area.get_corner(UR)
    return (
        max(uhy - ahy, 0.0),
        max(uhx - ahx, 0.0),
        max(aly - uly, 0.0),
        max(alx - ulx, 0.0),
    )


def _track_extents(
    layout: pd.DataFrame, ext: Mapping[int, Extents]
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Per column (left, right) and per row (top, bottom) max extents."""
    nrow, ncol = grid_shape(layout)
    left, right = [0.0] * ncol, [0.0] * ncol
    top, bottom = [0.0] * nrow, [0.0] * nrow
    for r in layout.itertuples(index=False):
        t, rt, b, lf = ext.get(int(r.PANEL), (0.0, 0.0, 0.0, 0.0))
        i, j = int(r.ROW) - 1, int(r.COL) - 1
        top[i], bottom[i] = max(top[i], t), max(bottom[i], b)
        left[j], right[j] = max(left[j], lf), max(right[j], rt)
    return left, right, top, bottom


def track_sizes(
    layout: pd.DataFrame,
    ext: Mapping[int, Extents],
    total: tuple[float, float],
    spacing: tuple[float, float],
    col_weights: Sequence[float] | None = None,
    row_weights: Sequence[float] | None = None,
    min_size: float = 0.3,
    strip_ext: Mapping[int, Mapping[str, float]] | None = None,
) -> tuple[list[float], list[float]]:
    """Plotting-area widths per column and heights per row.

    ``total`` (``theme.panel_size``) is the size of the grid *excluding* its
    outermost axis decorations (tick labels on the outer edge, as for a single
    panel) but *including* inner decorations, outer strips and ``spacing`` between
    tracks. It is shared by the tracks in proportion to ``col_weights`` /
    ``row_weights`` (default equal; ``facet_grid(space="free")`` passes range
    lengths). Each track is at least ``min_size``. ``strip_ext`` is
    ``{PANEL: {side: total strip thickness}}``: strips on the outer edge count
    against ``total`` (outer axis decorations do not).
    """
    nrow, ncol = grid_shape(layout)
    left, right, top, bottom = _track_extents(layout, ext)
    sx, sy = spacing
    # outer *axis* decorations do not count (a single panel's do not either);
    # outer strips do, so measure only what is inside the outer edges
    inner_w = sum(left[1:]) + sum(right[:-1]) + (ncol - 1) * sx
    inner_h = sum(top[1:]) + sum(bottom[:-1]) + (nrow - 1) * sy
    inner_w += _outer_strip(layout, strip_ext, "left") + _outer_strip(layout, strip_ext, "right")
    inner_h += _outer_strip(layout, strip_ext, "top") + _outer_strip(layout, strip_ext, "bottom")
    avail_w = max(total[0] - inner_w, min_size * ncol)
    avail_h = max(total[1] - inner_h, min_size * nrow)
    cw = np.asarray(col_weights if col_weights is not None else [1.0] * ncol, dtype=float)
    rw = np.asarray(row_weights if row_weights is not None else [1.0] * nrow, dtype=float)
    widths = [max(float(w), min_size) for w in avail_w * cw / cw.sum()]
    heights = [max(float(h), min_size) for h in avail_h * rw / rw.sum()]
    return widths, heights


def _outer_strip(
    layout: pd.DataFrame, strip_ext: Mapping[int, Mapping[str, float]] | None, side: str
) -> float:
    """Thickest strip on ``side`` among the panels on that outer edge."""
    if not strip_ext:
        return 0.0
    nrow, ncol = grid_shape(layout)
    best = 0.0
    for r in layout.itertuples(index=False):
        edge = {
            "top": int(r.ROW) == 1,
            "bottom": int(r.ROW) == nrow,
            "left": int(r.COL) == 1,
            "right": int(r.COL) == ncol,
        }[side]
        if edge:
            best = max(best, strip_ext.get(int(r.PANEL), {}).get(side, 0.0))
    return best


def place_units(
    layout: pd.DataFrame,
    units: Mapping[int, Mobject],
    areas: Mapping[int, Mobject],
    ext: Mapping[int, Extents],
    widths: Sequence[float],
    heights: Sequence[float],
    spacing: tuple[float, float],
) -> None:
    """Shift each ``units[PANEL]`` (panel + strips) so its plotting area
    ``areas[PANEL]`` is centred in its grid cell. Cells are laid out left to
    right / top to bottom with the per-track extents and ``spacing`` between
    tracks; the grid's top-left plotting-area corner is at the origin."""
    left, right, top, bottom = _track_extents(layout, ext)
    sx, sy = spacing
    xs, x = [], 0.0
    for j, w in enumerate(widths):
        if j:
            x += right[j - 1] + sx + left[j]
        xs.append(x)
        x += w
    ys, y = [], 0.0
    for i, h in enumerate(heights):
        if i:
            y -= bottom[i - 1] + sy + top[i]
        ys.append(y)
        y -= h
    for r in layout.itertuples(index=False):
        p = int(r.PANEL)
        i, j = int(r.ROW) - 1, int(r.COL) - 1
        cx = xs[j] + widths[j] / 2
        cy = ys[i] - heights[i] / 2
        lo, hi = area_box(areas[p])
        units[p].shift(np.array([cx - (lo[0] + hi[0]) / 2, cy - (lo[1] + hi[1]) / 2, 0.0]))
