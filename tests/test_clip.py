"""Unit tests for the pure-numpy clipping algorithms (ggmanim.geoms._clip)."""

from types import SimpleNamespace

import numpy as np

from ggmanim.geoms._clip import (
    clip_polygon,
    clip_polyline,
    clip_rects,
    clip_segments,
    inside,
    panel_rect,
)

R = (0.0, 10.0, 0.0, 5.0)


def test_inside_closed_and_nan():
    m = inside([0, 10, 5, -1, 11, 5, np.nan], [0, 5, 2.5, 1, 1, 6, 1], R)
    assert m.tolist() == [True, True, True, False, False, False, False]


def test_liang_barsky_basic_cases():
    # inside, crossing left edge, fully outside, parallel outside, both ends outside crossing
    x0 = np.array([1.0, -5.0, -5.0, 1.0, -5.0])
    y0 = np.array([1.0, 1.0, 10.0, 7.0, 2.5])
    x1 = np.array([2.0, 5.0, 15.0, 9.0, 15.0])
    y1 = np.array([2.0, 1.0, 10.0, 7.0, 2.5])
    t0, t1, vis = clip_segments(x0, y0, x1, y1, R)
    assert vis.tolist() == [True, True, False, False, True]
    np.testing.assert_allclose([t0[0], t1[0]], [0, 1])
    np.testing.assert_allclose([t0[1], t1[1]], [0.5, 1])
    np.testing.assert_allclose([t0[4], t1[4]], [0.25, 0.75])


def test_liang_barsky_diagonal_corner():
    t0, t1, vis = clip_segments(-1.0, -1.0, 11.0, 11.0, R)
    assert vis
    np.testing.assert_allclose([-1 + 12 * t0, -1 + 12 * t1], [0, 5])


def test_polyline_inside_unchanged():
    x, y = np.array([1.0, 2, 3]), np.array([1.0, 4, 2])
    ((px, py),) = clip_polyline(x, y, R)
    np.testing.assert_array_equal(px, x)
    np.testing.assert_array_equal(py, y)


def test_polyline_endpoints_on_border():
    x = np.linspace(-5, 15, 21)
    y = np.full_like(x, 2.0)
    ((px, py),) = clip_polyline(x, y, R)
    assert px[0] == 0.0 and px[-1] == 10.0
    np.testing.assert_allclose(py, 2.0)
    # interior vertices are kept exactly
    np.testing.assert_array_equal(px[1:-1], x[(x > 0) & (x < 10)])


def test_polyline_leaves_and_reenters():
    x = np.array([1.0, 5, 9])
    y = np.array([1.0, 9, 1])  # peaks above ymax=5
    pieces = clip_polyline(x, y, R)
    assert len(pieces) == 2
    (ax, ay), (bx, by) = pieces
    np.testing.assert_allclose([ax[0], ay[0], ax[-1], ay[-1]], [1, 1, 3, 5])
    np.testing.assert_allclose([bx[0], by[0], bx[-1], by[-1]], [7, 5, 9, 1])


def test_polyline_outside_and_degenerate():
    assert clip_polyline([20, 30], [20, 30], R) == []
    assert clip_polyline([], [], R) == []
    assert clip_polyline([20], [1], R) == []
    assert len(clip_polyline([2], [1], R)) == 1


def test_polygon_inside_unchanged_and_outside_empty():
    x, y = [1.0, 2, 2], [1.0, 1, 2]
    px, py = clip_polygon(x, y, R)
    np.testing.assert_array_equal(px, x)
    px, py = clip_polygon([20, 21, 21], [1, 1, 2], R)
    assert len(px) == 0


def test_polygon_sutherland_hodgman_square():
    # a square overhanging the bottom-left corner
    px, py = clip_polygon([-2, 2, 2, -2], [-2, -2, 2, 2], R)
    pts = {(round(a, 9), round(b, 9)) for a, b in zip(px, py)}
    assert pts == {(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)}
    assert inside(px, py, R).all()


def test_polygon_ribbon_area():
    # band y in [1, 9] over x in [-5, 15] clipped to [0,10]x[0,5] -> 10 x 4
    x = np.array([-5.0, 15, 15, -5])
    y = np.array([9.0, 9, 1, 1])
    px, py = clip_polygon(x, y, R)
    area = 0.5 * abs(np.dot(px, np.roll(py, 1)) - np.dot(py, np.roll(px, 1)))
    assert np.isclose(area, 40.0)


def test_clip_rects():
    xmin, xmax, ymin, ymax, vis = clip_rects([-1, 3, 20], [1, 4, 21], [0, -2, 0], [7, 2, 1], R)
    assert vis.tolist() == [True, True, False]
    np.testing.assert_allclose(xmin[:2], [0, 3])
    np.testing.assert_allclose(ymax[:2], [5, 2])
    np.testing.assert_allclose(ymin[:2], [0, 0])


def test_panel_rect():
    pp = SimpleNamespace(x_range=(0.0, 10.0), y_range=(-1.0, 1.0))
    assert panel_rect(SimpleNamespace(clip="on"), pp) == (0.0, 10.0, -1.0, 1.0)
    assert panel_rect(SimpleNamespace(clip="off"), pp) is None
    assert panel_rect(object(), pp) == (0.0, 10.0, -1.0, 1.0)  # default: on
    assert panel_rect(SimpleNamespace(clip="on"), None) is None
