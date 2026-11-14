"""coord_trans: transformed axes, input-space panel params, munched geoms (ggm-aji.1)."""

import numpy as np
import pandas as pd
import pytest
from manim import VMobject

from ggmanim import aes, coord_trans, ggplot, scale_x_continuous, scale_x_log10
from ggmanim.coords import is_linear
from ggmanim.coords_trans import CoordTrans, TransPanel
from ggmanim.registry import COMPONENTS
from ggmanim.scales.continuous import LogTransform


def _xy(m: VMobject) -> np.ndarray:
    return np.asarray(m.get_start_anchors())[:, :2]


def _line_df():
    return pd.DataFrame({"x": [1.0, 1000.0], "y": [1.0, 1000.0]})


def _straightness(pts: np.ndarray) -> float:
    """Largest distance of ``pts`` from the chord through its ends."""
    a, b = pts[0], pts[-1]
    d = b - a
    n = np.array([-d[1], d[0]]) / np.hypot(*d)
    return float(np.max(np.abs((pts - a) @ n)))


def test_registered_and_composes():
    assert "coord_trans" in COMPONENTS
    p = ggplot().coord_trans(x="log10", y="sqrt")
    assert isinstance(p.coord, CoordTrans)
    assert isinstance(p.coord.trans_x, LogTransform)
    assert p.coord.trans_y.name == "sqrt"
    assert not is_linear(p.coord)
    assert "coord_trans(x='log10', y='sqrt'" in repr(p.coord)


def test_unknown_transform_rejected():
    with pytest.raises(ValueError):
        coord_trans(x="nope")


def test_axes_over_transformed_range_and_input_space_params():
    p = ggplot(_line_df(), aes("x", "y")).geom_point().add(coord_trans(x="log10")).build()
    panel = p.panel
    assert isinstance(panel, TransPanel)
    # axes: expanded transformed range (log10 1..1000 = 0..3, 5% each side)
    np.testing.assert_allclose(panel.axes.x_range[:2], [-0.15, 3.15])
    np.testing.assert_allclose(panel.trans_range[0], [-0.15, 3.15])
    # panel params: input (data) space
    np.testing.assert_allclose(panel.panel_params.x_range, 10 ** np.array([-0.15, 3.15]))
    assert panel.panel_params.y_range == pytest.approx((1 - 49.95, 1000 + 49.95))


def test_breaks_from_scale_placed_at_transformed_positions():
    df = _line_df()
    p = (
        ggplot(df, aes("x", "y"))
        .geom_point()
        .add(coord_trans(x="log10"), scale_x_continuous(breaks=[1, 10, 100, 1000]))
        .build()
    )
    pp = p.panel_params
    assert pp.x_breaks == (1, 10, 100, 1000)
    assert pp.x_labels == ("1", "10", "100", "1000")
    xs = [lab.get_center()[0] for lab in p.panel.x_tick_labels]
    gaps = np.diff(xs)
    np.testing.assert_allclose(gaps, gaps[0], rtol=1e-6)  # equally spaced decades
    expect = p.panel.transform([1, 10, 100, 1000], 1)[:, 0]
    np.testing.assert_allclose(xs, expect, atol=1e-6)
    # vertical grid lines sit at the same positions
    vertical = [g.get_start()[0] for g in p.panel.grid_major if abs(g.get_vector()[0]) < 1e-9]
    np.testing.assert_allclose(sorted(vertical), expect, atol=1e-6)


def test_default_breaks_are_data_space_pretty():
    p = ggplot(_line_df(), aes("x", "y")).geom_point().add(coord_trans(x="log10")).build()
    b = np.asarray(p.panel_params.x_breaks)
    np.testing.assert_allclose(np.diff(b), np.diff(b)[0])  # pretty linear breaks
    xs = np.array([lab.get_center()[0] for lab in p.panel.x_tick_labels])
    assert np.all(np.diff(np.diff(xs)) < 0)  # ... crowded towards the right on log x


def test_points_at_transformed_positions():
    df = pd.DataFrame({"x": [1.0, 10.0, 100.0, 1000.0], "y": [1.0, 2.0, 3.0, 4.0]})
    p = ggplot(df, aes("x", "y")).geom_point().add(coord_trans(x="log10")).build()
    xs = np.array([m.get_center()[0] for m in p.layers[0].marks])
    np.testing.assert_allclose(np.diff(xs), np.diff(xs)[0], rtol=1e-6)


def test_straight_line_becomes_curve_with_many_vertices():
    p = ggplot(_line_df(), aes("x", "y")).geom_line().add(coord_trans(y="log10")).build()
    mark = p.layers[0].marks[0]
    pts = _xy(mark)
    assert len(pts) + 1 >= 20
    assert _straightness(np.vstack([pts, mark.get_end()[:2]])) > 0.5
    # every vertex is the transform of a point on the data-space line
    panel = p.panel
    xs = np.linspace(1, 1000, 2001)
    curve = panel.transform(xs, xs)[:, :2]
    for v in pts:
        assert np.min(np.hypot(*(curve - v).T)) < 0.02


def test_line_identical_to_cartesian_under_identity_coord_trans():
    df = pd.DataFrame({"x": [1.0, 3.0, 7.0], "y": [2.0, 5.0, 1.0]})
    a = ggplot(df, aes("x", "y")).geom_line().build().layers[0].marks[0]
    b = ggplot(df, aes("x", "y")).geom_line().add(coord_trans()).build().layers[0].marks[0]
    # munched, but every point still on the same straight segments
    pa, pb = _xy(a), _xy(b)
    assert len(pb) >= len(pa)
    for v in pa:
        assert np.min(np.hypot(*(pb - v).T)) < 1e-9


def test_lm_fit_in_data_space():
    rng = np.random.default_rng(0)
    x = np.linspace(1, 100, 40)
    df = pd.DataFrame({"x": x, "y": 3 + 2 * x + rng.normal(0, 5, 40)})
    plain = ggplot(df, aes("x", "y")).geom_smooth(method="lm").build()
    trans = ggplot(df, aes("x", "y")).geom_smooth(method="lm").add(coord_trans(x="log10")).build()
    a, b = list(plain.layer_data.values())[0], list(trans.layer_data.values())[0]
    for col in ("x", "y", "ymin", "ymax"):
        np.testing.assert_allclose(a[col].to_numpy(float), b[col].to_numpy(float))
    # the fitted straight line is drawn curved, and so is its ribbon
    ribbon, line = trans.layers[0].marks[0].submobjects
    assert _straightness(_xy(line)) > 0.3
    assert len(_xy(ribbon)) > 2 * len(b)


def test_lm_differs_from_scale_transform():
    x = np.linspace(1, 100, 30)
    df = pd.DataFrame({"x": x, "y": 2 * x})
    scaled = ggplot(df, aes("x", "y")).geom_smooth(method="lm").add(scale_x_log10()).build()
    d = list(scaled.layer_data.values())[0]
    assert d["x"].max() == pytest.approx(2.0)  # scale transform: fitted in log space
    coord = ggplot(df, aes("x", "y")).geom_smooth(method="lm").add(coord_trans(x="log10")).build()
    d2 = list(coord.layer_data.values())[0]
    assert d2["x"].max() == pytest.approx(100.0)


def test_rect_becomes_munched_polygon_with_expected_vertices():
    df = pd.DataFrame({"xmin": [1.0], "xmax": [100.0], "ymin": [1.0], "ymax": [100.0]})
    p = (
        ggplot(df)
        .geom_rect(aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"))
        .add(coord_trans(x="log10", y="sqrt"))
        .build()
    )
    m = p.layers[0].marks[0]
    pts = _xy(m)
    assert len(pts) > 4 and len(pts) % 4 == 0
    k = len(pts) // 4
    corners = p.panel.transform([1, 100, 100, 1], [1, 1, 100, 100])[:, :2]
    np.testing.assert_allclose(pts[::k], corners, atol=1e-9)
    # bottom edge: data-space x evenly spaced, drawn at log positions
    t = np.arange(k) / k
    np.testing.assert_allclose(pts[:k], p.panel.transform(1 + 99 * t, 1)[:, :2], atol=1e-9)


def test_col_from_zero_clamped_under_log_y():
    df = pd.DataFrame({"x": ["a", "b"], "y": [10.0, 1000.0]})
    p = ggplot(df, aes("x", "y")).geom_col().add(coord_trans(y="log10")).build()
    for m in p.layers[0].marks:
        pts = _xy(m)
        assert np.isfinite(pts).all()
        # bars reach the bottom of the panel (0 is below the log domain)
        bottom = p.panel.transform(1, p.panel_params.y_range[0])[0, 1]
        assert pts[:, 1].min() == pytest.approx(bottom)


def test_area_and_polygon_munched():
    df = pd.DataFrame({"x": [1.0, 1000.0], "y": [1000.0, 1000.0]})
    p = ggplot(df, aes("x", "y")).geom_area().add(coord_trans(x="log10")).build()
    assert len(_xy(p.layers[0].marks[0])) > 20
    tri = pd.DataFrame({"x": [1.0, 1000.0, 1.0], "y": [1.0, 1.0, 1000.0]})
    q = ggplot(tri, aes("x", "y")).geom_polygon().add(coord_trans(x="log10", y="log10")).build()
    pts = _xy(q.layers[0].marks[0])
    assert len(pts) > 20
    assert np.isfinite(pts).all()


def test_raster_falls_back_to_tiles_under_coord_trans():
    df = pd.DataFrame({"x": np.repeat([1.0, 2.0, 3.0], 3), "y": np.tile([1.0, 2.0, 3.0], 3)})
    df["z"] = np.arange(9.0)
    p = ggplot(df, aes("x", "y", fill="z")).geom_raster().add(coord_trans(x="log10")).build()
    assert len(p.layers[0].marks) == 9


def test_clipping_in_input_space():
    df = pd.DataFrame({"x": [1.0, 1000.0], "y": [1.0, 1000.0]})
    p = ggplot(df, aes("x", "y")).geom_line().add(coord_trans(x="log10", xlim=(10, 100))).build()
    pts = _xy(p.layers[0].marks[0])
    (x0, x1), _ = p.panel_params.x_range, p.panel_params.y_range
    lo = p.panel.transform(x0, 1)[0, 0]
    hi = p.panel.transform(x1, 1)[0, 0]
    assert pts[:, 0].min() >= lo - 1e-9 and pts[:, 0].max() <= hi + 1e-9


def test_xlim_zooms_in_data_space():
    p = (
        ggplot(_line_df(), aes("x", "y"))
        .geom_point()
        .add(coord_trans(x="log10", xlim=(10, 100), expand=False))
        .build()
    )
    np.testing.assert_allclose(p.panel_params.x_range, (10, 100))
    np.testing.assert_allclose(p.panel.trans_range[0], (1, 2))


def test_sqrt_expansion_below_zero_clamps_input_range():
    df = pd.DataFrame({"x": [0.0, 100.0], "y": [0.0, 1.0]})
    p = ggplot(df, aes("x", "y")).geom_point().add(coord_trans(x="sqrt")).build()
    lo, hi = p.panel_params.x_range
    assert lo == 0.0 and hi == pytest.approx(10.5**2)
    assert p.panel.trans_range[0][0] == pytest.approx(-0.5)


def test_reverse_transform():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    p = ggplot(df, aes("x", "y")).geom_point().add(coord_trans(x="reverse")).build()
    xs = [m.get_center()[0] for m in p.layers[0].marks]
    assert xs[0] > xs[1]
    lo, hi = p.panel_params.x_range
    assert lo < hi


def test_facets_use_trans_panels():
    df = pd.DataFrame(
        {"x": [1.0, 1000.0, 1.0, 1000.0], "y": [1.0, 2.0, 3.0, 4.0], "g": list("aabb")}
    )
    p = ggplot(df, aes("x", "y")).geom_line().facet_wrap("g").add(coord_trans(x="log10")).build()
    assert all(isinstance(pan, TransPanel) for pan in p.panels.values())
    for mob in p.layers[0].marks:
        assert len(_xy(mob)) > 20


def test_point_unaffected_by_munching():
    df = pd.DataFrame({"x": [1.0, 1000.0], "y": [1.0, 1000.0]})
    p = ggplot(df, aes("x", "y")).geom_point().add(coord_trans(x="log10")).build()
    assert len(p.layers[0].marks) == 2
