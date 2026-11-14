"""coord_munch / munch_path / munch_rect_corners (ggm-aji.1)."""

import numpy as np
import pytest

from ggmanim.coords import coord_cartesian, coord_munch, is_linear
from ggmanim.coords_trans import TransPanel, coord_trans
from ggmanim.geoms.base import munch_path, munch_rect_corners, transform_points
from ggmanim.scales import scale_x_continuous, scale_y_continuous
from ggmanim.theme import theme_default


def _scales(x=(1, 1000), y=(1, 1000)):
    sx, sy = scale_x_continuous(), scale_y_continuous()
    sx.train(list(x))
    sy.train(list(y))
    return sx, sy


def _panel(coord, x=(1, 1000), y=(1, 1000)):
    return coord.build(*_scales(x, y), theme_default())


def test_is_linear_flags():
    assert is_linear(coord_cartesian())
    assert is_linear(_panel(coord_cartesian()))
    assert not is_linear(coord_trans(x="log10"))
    panel = _panel(coord_trans(x="log10"))
    assert isinstance(panel, TransPanel) and not is_linear(panel)
    assert is_linear(object())  # anything without the flag is linear


def test_linear_coord_is_untouched():
    panel = _panel(coord_cartesian())
    x, y = coord_munch(panel, [1, 1000, 500], [1, 1000, 2])
    np.testing.assert_array_equal(x, [1, 1000, 500])
    np.testing.assert_array_equal(y, [1, 1000, 2])
    corners = munch_rect_corners(panel, [[1, 10, 10, 1]], [[1, 1, 10, 10]])
    assert corners.shape == (1, 4, 3)


def test_munch_keeps_vertices_and_interpolates_linearly_in_input_space():
    panel = _panel(coord_trans(y="log10"))
    x, y = munch_path(panel, [0, 1000], [1, 1000])
    assert len(x) >= 20
    assert (x[0], y[0]) == (0, 1) and (x[-1], y[-1]) == (1000, 1000)
    # every new vertex lies on the straight input-space segment
    np.testing.assert_allclose(y, 1 + (x - 0) * 999 / 1000)
    assert np.all(np.diff(x) > 0)


def test_segment_length_is_fiftieth_of_panel():
    panel = _panel(coord_trans(x="log10"))
    (t0, t1), (u0, u1) = panel.trans_range
    x, y = munch_path(panel, [1, 1000], [500, 500])
    # transformed length of the segment as a fraction of the panel width
    frac = (np.log10(1000) - np.log10(1)) / (t1 - t0)
    assert len(x) - 1 == int(np.ceil(frac * 50))
    # pieces are equal in input space (as in ggplot2), not in transformed space
    np.testing.assert_allclose(np.diff(x), 999 / (len(x) - 1))


def test_closed_ring_munches_closing_edge():
    panel = _panel(coord_trans(x="log10", y="log10"))
    x, y = munch_path(panel, [1, 1000, 1], [1, 1, 1000], closed=True)
    assert (x[0], y[0]) == (1, 1)
    assert not (x[-1] == 1 and y[-1] == 1)  # first vertex not repeated
    # closing edge (1, 1000) -> (1, 1) is vertical: interpolated points have x == 1
    tail = x[np.argmax((x == 1) & (y == 1000)) :]
    assert np.all(tail == 1) and len(tail) > 2


def test_nan_segments_are_not_subdivided():
    panel = _panel(coord_trans(y="log10"))
    x, y = munch_path(panel, [1, 2, np.nan, 900, 1000], [1, 1000, np.nan, 1, 1000])
    assert np.isnan(x).sum() == 1
    i = int(np.flatnonzero(np.isnan(x))[0])
    assert x[i - 1] == 2 and x[i + 1] == 900  # path break kept between real vertices


def test_out_of_domain_vertex_is_kept():
    panel = _panel(coord_trans(y="log10"))
    x, y = munch_path(panel, [1, 10, 100], [0, 10, 100])  # log10(0) undefined
    # ggm-78e: drawn at the panel edge, so munched from that edge's input value
    assert y[0] == panel.panel_params.y_range[0] and x[0] == 1
    assert len(x) > 3


def test_munched_points_follow_the_transform():
    panel = _panel(coord_trans(x="log10"))
    x, y = munch_path(panel, [1, 1000], [1, 1000])
    pts = panel.transform(x, y)
    # scene x is linear in log10(x); scene y linear in y
    sx = np.polyfit(np.log10(x), pts[:, 0], 1)
    sy = np.polyfit(y, pts[:, 1], 1)
    np.testing.assert_allclose(np.polyval(sx, np.log10(x)), pts[:, 0], atol=1e-9)
    np.testing.assert_allclose(np.polyval(sy, y), pts[:, 1], atol=1e-9)
    # ... so the straight data-space line is a curve on screen
    chord = np.polyfit(pts[[0, -1], 0], pts[[0, -1], 1], 1)
    assert np.max(np.abs(np.polyval(chord, pts[:, 0]) - pts[:, 1])) > 0.5


def test_rect_becomes_munched_four_sided_polygon():
    panel = _panel(coord_trans(x="log10", y="log10"))
    cx = np.array([[1, 1000, 1000, 1], [10, 100, 100, 10]], dtype=float)
    cy = np.array([[1, 1, 1000, 1000], [10, 10, 100, 100]], dtype=float)
    rings = munch_rect_corners(panel, cx, cy)
    n, m, _ = rings.shape
    assert n == 2 and m % 4 == 0 and m > 4
    k = m // 4
    # each edge starts at its corner: the 4 corners are at indices 0, k, 2k, 3k
    expect = transform_points(panel, cx.ravel(), cy.ravel()).reshape(2, 4, 3)
    np.testing.assert_allclose(rings[:, ::k], expect)
    # bottom edge vertices: y constant, x interpolated linearly in data space
    t = np.arange(k) / k
    want = transform_points(panel, 1 + 999 * t, np.ones(k))
    np.testing.assert_allclose(rings[0, :k], want)
    # 4 sides: under log x/log y every edge stays axis-aligned on screen
    for side in range(4):
        e = np.vstack([rings[0, side * k : (side + 1) * k], rings[0, ((side + 1) * k) % m]])
        assert np.ptp(e[:, 0]) < 1e-9 or np.ptp(e[:, 1]) < 1e-9


def test_rect_corners_empty():
    panel = _panel(coord_trans(x="log10"))
    assert munch_rect_corners(panel, np.zeros((0, 4)), np.zeros((0, 4))).shape[0] == 0


@pytest.mark.parametrize("segments", [10, 100])
def test_segments_parameter(segments):
    panel = _panel(coord_trans(x="log10"))
    x, _ = coord_munch(panel, [1, 1000], [1, 1], segments=segments)
    (t0, t1), _ = panel.trans_range
    assert len(x) - 1 == int(np.ceil(3 / (t1 - t0) * segments))
