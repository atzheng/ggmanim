"""coord_cartesian zoom vs scale limits, oob, clipping, coord_flip, coord_fixed."""

import warnings

import numpy as np
import pandas as pd
import pytest

import ggmanim
from ggmanim import aes, ggplot
from ggmanim.coords import CoordFixed, CoordFlip, coord_cartesian, coord_fixed, coord_flip
from ggmanim.registry import COMPONENTS
from ggmanim.scales import scale_x_continuous, xlim, ylim


@pytest.fixture
def curved():
    x = np.linspace(0, 10, 41)
    return pd.DataFrame({"x": x, "y": (x - 5) ** 2})


def _panel_box(m, tol=1e-6):
    pp = m.panel_params
    corners = m.panel.transform([pp.x_range[0], pp.x_range[1]], [pp.y_range[0], pp.y_range[1]])
    lo = corners.min(axis=0) - tol
    hi = corners.max(axis=0) + tol
    return lo, hi


def _points(mob):
    pts = [np.asarray(m.points) for m in mob.family_members_with_points()]
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def _inside_box(pts, box):
    lo, hi = box
    return np.all((pts[:, :2] >= lo[:2]) & (pts[:, :2] <= hi[:2]))


# -- zoom vs censor ---------------------------------------------------------------


def test_zoomed_smooth_uses_all_data_and_touches_both_edges(curved):
    m = ggplot(curved, aes("x", "y")).geom_smooth(se=False).coord_cartesian(xlim=(3, 7)).build()
    fit = m.layer_data[0]
    slope, icpt = np.polyfit(curved["x"], curved["y"], 1)
    np.testing.assert_allclose(fit["y"], slope * fit["x"] + icpt, atol=1e-8)
    line = m.layers[0][0]
    pts = _points(line)
    lo, hi = _panel_box(m)
    assert np.isclose(pts[:, 0].min(), lo[0], atol=1e-5)
    assert np.isclose(pts[:, 0].max(), hi[0], atol=1e-5)
    assert _inside_box(pts, (lo, hi))


def test_censored_smooth_fits_in_range_only_and_stops_short(curved):
    with pytest.warns(UserWarning, match="removed"):
        m = ggplot(curved, aes("x", "y")).geom_smooth(se=False).add(xlim(3, 7)).build()
    fit = m.layer_data[0]
    keep = (curved["x"] >= 3) & (curved["x"] <= 7)
    slope, icpt = np.polyfit(curved["x"][keep], curved["y"][keep], 1)
    np.testing.assert_allclose(fit["y"], slope * fit["x"] + icpt, atol=1e-8)
    assert fit["x"].min() == pytest.approx(3) and fit["x"].max() == pytest.approx(7)
    pts = _points(m.layers[0][0])
    lo, hi = _panel_box(m)
    assert pts[:, 0].min() > lo[0] + 0.05 and pts[:, 0].max() < hi[0] - 0.05


def test_zoom_does_not_warn(curved):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ggplot(curved, aes("x", "y")).geom_point().coord_cartesian(xlim=(3, 7)).build()


# -- oob policies & warnings -------------------------------------------------------


def test_censor_warns_once_per_layer_with_count(curved):
    n_out = int(((curved["x"] < 3) | (curved["x"] > 7)).sum())
    with pytest.warns(UserWarning) as rec:
        m = ggplot(curved, aes("x", "y")).geom_point().geom_line().add(xlim(3, 7)).build()
    msgs = [str(w.message) for w in rec if "removed" in str(w.message)]
    assert len(msgs) == 2
    assert all(f"removed {n_out} rows" in s for s in msgs)
    assert "layer 0 (GeomPoint)" in msgs[0] and "layer 1 (GeomLine)" in msgs[1]
    assert len(m.layers[0]) == len(curved) - n_out


def test_na_rm_silences_warning(curved):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = ggplot(curved, aes("x", "y")).geom_point(na_rm=True).add(xlim(3, 7)).build()
    assert len(m.layers[0]) == int(((curved["x"] >= 3) & (curved["x"] <= 7)).sum())


def test_missing_values_dropped_with_warning():
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0], "y": [1.0, np.nan, 3.0, 4.0]})
    with pytest.warns(UserWarning, match="removed 2 rows"):
        m = ggplot(df, aes("x", "y")).geom_point().build()
    assert len(m.layers[0]) == 2


def test_oob_squish_clamps_before_stat(curved):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = (
            ggplot(curved, aes("x", "y"))
            .geom_point()
            .add(scale_x_continuous(limits=(3, 7), oob="squish"))
            .build()
        )
    d = m.layer_data[0]
    assert len(m.layers[0]) == len(curved)
    assert d["x"].min() == 3 and d["x"].max() == 7


def test_oob_keep_keeps_data_but_clip_hides_marks(curved):
    m = (
        ggplot(curved, aes("x", "y"))
        .geom_point()
        .add(scale_x_continuous(limits=(3, 7), oob="keep"))
        .build()
    )
    d = m.layer_data[0]
    assert d["x"].min() == 0 and d["x"].max() == 10  # kept, not censored
    assert _inside_box(_points(m.layers[0]), _panel_box(m, tol=0.2))


# -- clipping ---------------------------------------------------------------------


def _zoomed_plot(df, clip="on"):
    bars = pd.DataFrame({"x": [1.0, 5.0, 9.0], "y": [30.0, 5.0, -3.0]})
    return (
        ggplot(df, aes("x", "y"))
        .geom_point()
        .geom_line()
        .geom_smooth(method="lm", degree=2)
        .geom_col(data=bars)
        .geom_vline(xintercept=[1, 5])
        .geom_hline(yintercept=[2, 40])
        .coord_cartesian(xlim=(3, 7), ylim=(0, 10), clip=clip)
    )


def test_zoomed_render_has_no_marks_outside_panel(curved):
    m = _zoomed_plot(curved).build()
    box = _panel_box(m)
    radius = 0.07  # dots are drawn around in-panel centres
    for key, layer in m.layers.items():
        pts = _points(layer)
        if len(pts) == 0:
            continue
        lo, hi = box
        ok = (pts[:, :2] >= lo[:2] - radius) & (pts[:, :2] <= hi[:2] + radius)
        assert ok.all(), f"layer {key} draws outside the panel"
    # out-of-panel reference lines are dropped
    assert len(m.layers[4]) == 1 and len(m.layers[5]) == 1


def test_clip_off_draws_outside(curved):
    m = _zoomed_plot(curved, clip="off").build()
    assert not _inside_box(_points(m.layers[1]), _panel_box(m))
    assert len(m.layers[4]) == 2


def test_clipped_polyline_endpoints_on_border(curved):
    m = ggplot(curved, aes("x", "y")).geom_line().coord_cartesian(xlim=(2, 8)).build()
    path = m.layers[0][0]
    lo, hi = _panel_box(m)
    start, end = path.get_start(), path.get_end()
    assert np.isclose(start[0], lo[0], atol=1e-6) or np.isclose(start[1], hi[1], atol=1e-6)
    assert np.isclose(end[0], hi[0], atol=1e-6) or np.isclose(end[1], hi[1], atol=1e-6)


def test_path_split_into_subpaths_when_leaving_panel():
    df = pd.DataFrame({"x": [0.0, 5.0, 10.0], "y": [0.0, 10.0, 0.0]})
    m = ggplot(df, aes("x", "y")).geom_line().coord_cartesian(ylim=(0, 4)).build()
    path = m.layers[0][0]
    assert len(path.get_subpaths()) == 2
    assert _inside_box(_points(path), _panel_box(m))


def test_dashed_line_clipped_then_dashed():
    # the peak leaves the zoomed panel at x=2 and re-enters at x=8 (y=4)
    df = pd.DataFrame({"x": [0.0, 5.0, 10.0], "y": [0.0, 10.0, 0.0]})
    m = (
        ggplot(df, aes("x", "y"))
        .geom_line(linetype="dashed")
        .coord_cartesian(ylim=(0, 4), expand=False)
        .build()
    )
    dashes = m.layers[0][0]
    assert len(dashes.submobjects) > 4  # dashed, not one solid path
    pts = _points(dashes)
    assert _inside_box(pts, _panel_box(m))
    gap_lo, gap_hi = m.panel.transform([2.0, 8.0], [4.0, 4.0])[:, 0]
    tol = 1e-6
    for d in dashes.submobjects:
        dx = np.asarray(d.points)[:, 0]
        # no dash spans or enters the part of the path outside the panel
        assert dx.max() <= gap_lo + tol or dx.min() >= gap_hi - tol
    xs = pts[:, 0]
    assert (xs < gap_lo).any() and (xs > gap_hi).any()  # dashes on both sides


def test_unzoomed_line_keeps_every_vertex(curved):
    m = ggplot(curved, aes("x", "y")).geom_line().build()
    assert len(m.layers[0][0].get_anchors_and_handles()[0]) == len(curved) - 1


def test_clip_validated():
    with pytest.raises(ValueError, match="clip"):
        coord_cartesian(clip="maybe")


# -- coord_flip -------------------------------------------------------------------


def test_coord_flip_registered_exported():
    for name in ("coord_flip", "coord_fixed"):
        assert name in COMPONENTS and name in ggmanim.__all__
    assert isinstance(ggplot().coord_flip().coord, CoordFlip)
    assert isinstance(ggplot().coord_fixed(2).coord, CoordFixed)


def test_coord_flip_puts_x_on_vertical_axis():
    df = pd.DataFrame({"x": [0.0, 5.0, 10.0], "y": [100.0, 200.0, 150.0]})
    m = ggplot(df, aes("x", "y")).geom_point().coord_flip().build()
    centres = np.array([d.get_center() for d in m.layers[0]])
    # x column increases -> scene y increases; y column ordering -> scene x
    assert np.all(np.diff(centres[:, 1]) > 0)
    assert np.argsort(centres[:, 0]).tolist() == np.argsort(df["y"].to_numpy()).tolist()
    # panel_params stay in aesthetic orientation
    assert m.panel_params.x_range[0] < 0 < 10 < m.panel_params.x_range[1]
    assert m.panel_params.y_range[0] < 100
    # the horizontal axis titles/labels are the y aesthetic's
    assert m.panel.x_title.text == "y" and m.panel.y_title.text == "x"
    assert m.panel.x_tick_labels[0].get_center()[1] < m.panel.y_tick_labels[0].get_center()[1]


def test_coord_flip_bars_are_horizontal():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 6.0, 9.0]})
    m = ggplot(df, aes("x", "y")).geom_col().coord_flip().build()
    widths = [b.width for b in m.layers[0]]
    heights = [b.height for b in m.layers[0]]
    assert np.all(np.diff(widths) > 0)
    np.testing.assert_allclose(heights, heights[0], rtol=1e-6)


def test_coord_flip_vline_is_horizontal():
    df = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 1.0]})
    m = ggplot(df, aes("x", "y")).geom_point().geom_vline(xintercept=5).coord_flip().build()
    line = m.layers[1][0]
    assert line.height < 1e-6 and line.width > 1


def test_coord_flip_xlim_refers_to_x_aesthetic():
    df = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 1.0]})
    m = ggplot(df, aes("x", "y")).geom_point().coord_flip(xlim=(2, 4), expand=False).build()
    assert m.panel_params.x_range == pytest.approx((2, 4))
    assert m.panel.axes.y_range[:2] == pytest.approx([2, 4])


# -- coord_fixed ------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0])
def test_coord_fixed_aspect(ratio):
    df = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 3.0]})
    m = ggplot(df, aes("x", "y")).geom_point().coord_fixed(ratio).build()
    o, px, py = m.panel.transform([0, 1, 0], [0, 0, 1])
    unit_x, unit_y = np.linalg.norm(px - o), np.linalg.norm(py - o)
    assert unit_y / unit_x == pytest.approx(ratio)


def test_coord_fixed_fits_theme_panel_size():
    from ggmanim.theme import theme_default

    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 100.0]})
    t = theme_default()
    w, h = t.panel_size()
    panel = coord_fixed(0.05).build(*_trained(df), t)
    assert panel.axes.x_length <= w + 1e-9 and panel.axes.y_length <= h + 1e-9
    assert max(panel.axes.x_length / w, panel.axes.y_length / h) == pytest.approx(1.0)


def _trained(df):
    sx, sy = scale_x_continuous(), ylim(None, None)
    sx.train(df["x"])
    sy.train(df["y"])
    return sx, sy


def test_coord_flip_theme_switches_follow_screen_axes():
    # ggplot2: *_x theme elements style the horizontal axis even under coord_flip
    from ggmanim.theme import theme

    df = pd.DataFrame({"x": [0.0, 10.0], "y": [100.0, 300.0]})
    p = ggplot(df, aes("x", "y")).geom_point().coord_flip()
    m = p.add(theme(axis_text_x=False)).build()
    assert len(m.panel.x_tick_labels) == 0 and len(m.panel.y_tick_labels) > 0
    m = p.add(theme(axis_text_x_angle=45)).build()
    labs = m.panel.x_tick_labels  # the y aesthetic's labels, on the horizontal axis
    assert [t.text for t in labs] == list(m.panel_params.y_labels)
    axis_y = m.panel.transform(m.panel_params.x_range[0], m.panel_params.y_range[0])[0, 1]
    for t in labs:  # rotated and hanging below the horizontal axis
        assert t.get_top()[1] < axis_y and t.width > 0 and t.height > 0


def _corners(mob):
    pts = np.asarray(mob.get_all_points())
    return pts[:, :2].min(axis=0), pts[:, :2].max(axis=0)


def test_background_and_border_use_coord_fixed_size_and_survive_clipping():
    from ggmanim.theme import theme

    df = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 3.0]})
    th = theme(panel_background="#222222", panel_border=True)
    for coord in (coord_fixed(1.0, xlim=(2, 8)), coord_flip(ylim=(1, 2))):
        m = ggplot(df, aes("x", "y")).geom_line().add(coord).add(th).build()
        assert m.panel.background is not None and m.panel.border is not None
        assert m.panel.background in m.panel.submobjects
        assert m.panel.border in m.panel.submobjects
        lo, hi = _panel_box(m)
        for deco in (m.panel.background, m.panel.border):
            dlo, dhi = _corners(deco)
            np.testing.assert_allclose(dlo, lo[:2], atol=1e-5)
            np.testing.assert_allclose(dhi, hi[:2], atol=1e-5)
    # the background has coord_fixed's aspect, not theme.panel_size's
    m = ggplot(df, aes("x", "y")).geom_line().coord_fixed(1.0).add(th).build()
    w, h = m.panel.background.width, m.panel.background.height
    pp = m.panel_params
    assert h / w == pytest.approx(np.ptp(pp.y_range) / np.ptp(pp.x_range))


def test_coord_flip_repr_and_fixed_validation():
    assert repr(coord_flip()).startswith("coord_flip(")
    with pytest.raises(ValueError):
        coord_fixed(0)
