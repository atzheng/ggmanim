"""Marching squares: analytic fields, orientation, saddles, bands, contour stats."""

import numpy as np
import pandas as pd
import pytest

from ggmanim import aes, geom_contour, geom_contour_filled, ggplot
from ggmanim.stats import StatContour, StatContourFilled, contour_breaks, isoband_labels
from ggmanim.stats._marching_squares import SEGMENT_TABLE, grid_from_xyz, isoband, isolines

X = np.linspace(-2, 2, 41)  # cell size 0.1
XX, YY = np.meshgrid(X, X)
R = np.hypot(XX, YY)


def signed_area(ring):
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def winding(px, py, rings):
    """Nonzero winding number of (px, py) w.r.t. the closed rings."""
    w = 0
    for r in rings:
        x, y = r[:, 0], r[:, 1]
        x2, y2 = np.roll(x, -1), np.roll(y, -1)
        for a, b, c, d in zip(x, y, x2, y2):
            cross = (c - a) * (py - b) - (px - a) * (d - b)
            if b <= py < d and cross > 0:
                w += 1
            elif d <= py < b and cross < 0:
                w -= 1
    return w


@pytest.mark.parametrize("r", [0.5, 1.0, 1.5, 1.93])
def test_circle_contours_within_one_cell(r):
    lines = isolines(X, X, -R, -r)  # superlevel set of -R: the disc
    assert len(lines) == 1
    line = lines[0]
    assert np.allclose(line[0], line[-1])  # closed
    assert np.abs(np.hypot(line[:, 0], line[:, 1]) - r).max() < 0.1
    # counter-clockwise around the inside (the disc), area close to pi r^2
    assert signed_area(line[:-1]) == pytest.approx(np.pi * r**2, rel=0.02)


def test_orientation_holes_clockwise():
    # superlevel set of R is outside the circle: the contour runs clockwise
    (line,) = isolines(X, X, R, 1.0)
    assert signed_area(line[:-1]) < 0


def test_open_lines_stop_at_border():
    lines = isolines(X, X, XX + YY, 0.5)
    assert len(lines) == 1
    line = lines[0]
    assert not np.allclose(line[0], line[-1])
    assert np.allclose(line.sum(axis=1), 0.5)
    border = np.isclose(np.abs(line[[0, -1]]), 2).any(axis=1)
    assert border.all()


def test_segment_table_edges():
    # every case's segments start and end on edges that separate inside from outside
    corners_of_edge = {0: (0, 1), 1: (1, 2), 2: (3, 2), 3: (0, 3)}
    for case, entries in SEGMENT_TABLE.items():
        bits = [(case >> k) & 1 for k in range(4)]  # bl, br, tr, tl
        crossed = {e for e, (a, b) in corners_of_edge.items() if bits[a] != bits[b]}
        pairs = [p for alt in entries for p in alt] if case in (5, 10) else list(entries)
        used = {e for p in pairs for e in p}
        assert used == crossed


def test_saddle_disambiguation():
    # one saddle cell: inside corners bottom-left and top-right; average 0.5
    Z = np.array([[1.0, 0.0], [0.0, 1.0]])
    x = np.array([0.0, 1.0])
    # level 0.4 <= average: inside corners connect, segments cut off the outside corners
    near = sorted(tuple(np.round(seg.mean(axis=0), 2)) for seg in isolines(x, x, Z, 0.4))
    assert near == [(0.2, 0.8), (0.8, 0.2)]
    # level 0.6 > average: segments cut off the inside corners
    near = sorted(tuple(np.round(seg.mean(axis=0), 2)) for seg in isolines(x, x, Z, 0.6))
    assert near == [(0.2, 0.2), (0.8, 0.8)]


def test_missing_cells_are_skipped():
    Z = -R.copy()
    Z[20, 20] = np.nan  # hole at the centre does not affect r=1 contour
    (line,) = isolines(X, X, Z, -1.0)
    assert np.allclose(line[0], line[-1])
    Z[20, 30] = np.nan  # on the r=1 circle: the line opens
    lines = isolines(X, X, Z, -1.0)
    assert all(not np.allclose(ln[0], ln[-1]) for ln in lines)


def test_band_area_and_winding():
    rings = isoband(X, X, R, 1.0, 1.5)
    area = sum(signed_area(r) for r in rings)
    assert area == pytest.approx(np.pi * (1.5**2 - 1.0), rel=0.01)
    assert winding(1.2, 0.05, rings) == 1
    assert winding(0.1, 0.1, rings) == 0  # inside the hole
    assert winding(1.9, 1.9, rings) == 0  # beyond 1.5
    # disc band: lowest band includes the grid border
    rings = isoband(X, X, -R, -10, -1.0)
    assert winding(1.95, 1.95, rings) == 1 and winding(0.0, 0.0, rings) == 0
    # band above the maximum is empty
    assert isoband(X, X, -R, 1.0, 2.0) == []


def test_band_cancels_shared_border():
    # whole border above hi: the border ring of lo and hi cancel out (2 rings left)
    rings = isoband(X, X, R, 1.0, 1.5)
    assert len(rings) == 2


def test_grid_from_xyz_incomplete():
    ux, uy, Z = grid_from_xyz([0, 1, 0], [0, 0, 1], [1, 2, 3])
    assert ux.tolist() == [0, 1] and uy.tolist() == [0, 1]
    assert Z[0].tolist() == [1, 2] and Z[1, 0] == 3 and np.isnan(Z[1, 1])


# -- breaks ------------------------------------------------------------------------


def test_contour_breaks_rules():
    assert contour_breaks((0.03, 0.97)).tolist() == pytest.approx(np.arange(0, 1.01, 0.1))
    assert contour_breaks((0, 1), breaks=[0.2, 0.7]).tolist() == [0.2, 0.7]
    assert contour_breaks((0.0, 1.0), binwidth=0.25).tolist() == [0, 0.25, 0.5, 0.75, 1.0]
    b = contour_breaks((0.0, 1.0), bins=5)
    assert len(b) >= 6 and b[0] <= 0 and b[-1] >= 1
    assert contour_breaks((0.0, 1.0), bins=1).tolist() == [0.0, 1.0]


def test_isoband_labels_like_r_format():
    assert isoband_labels([0, 0.125, 0.25]) == ["(0.000, 0.125]", "(0.125, 0.250]"]
    assert isoband_labels([0, 0.02, 0.04]) == ["(0.00, 0.02]", "(0.02, 0.04]"]
    assert isoband_labels([0, 5, 10]) == ["(0, 5]", "(5, 10]"]


# -- stats / geoms ---------------------------------------------------------------------


def grid_df():
    return pd.DataFrame({"x": XX.ravel(), "y": YY.ravel(), "z": np.exp(-(R**2)).ravel()})


def test_stat_contour_output():
    out = StatContour(breaks=[0.25, 0.5, 0.75]).compute(grid_df())
    assert set(out.columns) >= {"x", "y", "level", "nlevel", "piece", "group"}
    assert sorted(out["level"].unique()) == [0.25, 0.5, 0.75]
    assert out["group"].nunique() == 3  # one closed ring per level
    assert out["nlevel"].max() == pytest.approx(1.0)
    for lev, gdf in out.groupby("level"):
        r = np.sqrt(-np.log(lev))
        assert np.abs(np.hypot(gdf["x"], gdf["y"]) - r).max() < 0.1


def test_stat_contour_filled_output():
    out = StatContourFilled(breaks=[0.0, 0.5, 1.0]).compute(grid_df())
    assert isinstance(out["level"].dtype, pd.CategoricalDtype)
    assert list(out["level"].cat.categories) == ["(0.0, 0.5]", "(0.5, 1.0]"]
    assert out["level"].cat.ordered
    assert out["group"].nunique() == 2
    assert {"subgroup", "level_low", "level_high", "level_mid"} <= set(out.columns)


def test_geoms_build_and_marks():
    g = (
        ggplot(grid_df(), aes("x", "y", z="z"))
        .add(geom_contour_filled(bins=4), geom_contour(bins=4))
        .build()
    )
    bands, lines = g.layers[0], g.layers[1]
    assert len(bands.marks) >= 3 and len(lines.marks) >= 3
    assert all(m.get_fill_opacity() == pytest.approx(1.0) for m in bands.marks)
