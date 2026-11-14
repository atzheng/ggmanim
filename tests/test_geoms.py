"""Structural tests for geoms: build real panels and inspect the drawn mobjects."""

import warnings

import numpy as np
import pandas as pd
import pytest
from manim import BLUE, RED, VGroup, VMobject

from ggmanim.aes import aes, evaluate_aes
from ggmanim.coords import CoordCartesian
from ggmanim.geoms import (
    KEY_SIZE,
    Geom,
    GeomCol,
    GeomFunction,
    GeomHline,
    GeomLine,
    GeomPoint,
    GeomSmooth,
    GeomVline,
    geom_bar,
    geom_function,
    geom_hline,
    geom_line,
    geom_point,
    geom_smooth,
    geom_vline,
    resolution,
)
from ggmanim.layer import Layer
from ggmanim.plot import ggplot
from ggmanim.registry import COMPONENTS
from ggmanim.scales import scale_x_continuous, scale_x_log10, scale_y_continuous
from ggmanim.scales.continuous import X_AESTHETICS, Y_AESTHETICS
from ggmanim.scales.discrete import ScaleDiscrete
from ggmanim.stats import StatFunction, StatIdentity, StatSmooth, add_group
from ggmanim.theme import theme_default

GEOMS = [
    "geom_point",
    "geom_line",
    "geom_col",
    "geom_bar",
    "geom_vline",
    "geom_hline",
    "geom_function",
    "geom_smooth",
]


def _panel(x, y, xs=scale_x_continuous, ys=scale_y_continuous):
    sx, sy = xs(), ys()
    sx.train(np.asarray(x, dtype=float))
    sy.train(np.asarray(y, dtype=float))
    return sx, sy, CoordCartesian().build(sx, sy, theme_default())


def _vertices(m: VMobject) -> np.ndarray:
    return np.vstack([m.get_start_anchors(), m.get_end_anchors()[-1:]])


def _color(m: VMobject) -> str:
    return m.get_stroke_color().to_hex().upper()


# -- construction / registry ------------------------------------------------------


@pytest.mark.parametrize("name", GEOMS)
def test_registered(name):
    assert name in COMPONENTS


def test_constructors_return_layers_and_chain():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 1, 2]})
    p = (
        ggplot(df, aes("a", "b"))
        .geom_point(color=RED, name="pts")
        .geom_line()
        .geom_col(width=0.5)
        .geom_vline(xintercept=2)
        .geom_function(np.sin)
        .geom_smooth(se=False)
    )
    assert all(isinstance(lyr, Layer) for lyr in p.layers)
    kinds = [type(lyr.geom) for lyr in p.layers]
    assert kinds == [GeomPoint, GeomLine, GeomCol, GeomVline, GeomFunction, GeomSmooth]
    stats = [type(lyr.stat) for lyr in p.layers]
    assert stats == [
        StatIdentity,
        StatIdentity,
        StatIdentity,
        StatIdentity,
        StatFunction,
        StatSmooth,
    ]
    assert p.layer("pts").params == {"color": RED}
    assert p.layers[5].stat.params["se"] is False and p.layers[5].params["se"] is False
    assert p.layers[4].stat.params["fun"] is np.sin and not p.layers[4].inherit_aes


def test_geom_bar_is_col_alias():
    assert isinstance(geom_bar().geom, GeomCol)


def test_param_aliases_and_unknown_warning():
    layer = geom_line(colour="red", stroke_width=6)
    assert layer.params == {"color": "red", "linewidth": 6}
    with pytest.warns(UserWarning, match="unknown parameters: bogus"):
        geom_point(bogus=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        geom_smooth(degree=2, level=0.9, fill="grey")
        geom_line(label="a")


def test_mapping_data_order_and_legacy_function_kwarg():
    df = pd.DataFrame({"a": [1]})
    layer = geom_point(df, aes("a", "a"))
    assert layer.data is df and layer.mapping == aes("a", "a")
    assert geom_function(function=np.cos).stat.params["fun"] is np.cos
    with pytest.raises(TypeError):
        geom_function()


def test_vline_makes_own_data():
    layer = geom_vline(xintercept=[1, 3], color=RED)
    assert not layer.inherit_aes
    assert list(layer.data["xintercept"]) == [1.0, 3.0]
    assert layer.mapping == aes(xintercept="xintercept")
    assert "xintercept" not in layer.params
    with pytest.raises(TypeError):
        geom_hline(aes(yintercept="a"), yintercept=1)


def test_position_aes_for_training():
    df = pd.DataFrame({"x": [1.0], "y": [2.0], "ymin": [0.0], "ymax": [3.0], "group": [-1]})
    assert GeomSmooth().position_aes_for_training(df) == {"x": ["x"], "y": ["y", "ymin", "ymax"]}
    assert GeomPoint().position_aes_for_training(df) == {"x": ["x"], "y": ["y", "ymin", "ymax"]}
    vdf = pd.DataFrame({"xintercept": [3.0], "x": [9.0]})
    assert GeomVline().position_aes_for_training(vdf) == {"x": ["xintercept"], "y": []}
    hdf = pd.DataFrame({"yintercept": [3.0], "y": [9.0]})
    assert GeomHline().position_aes_for_training(hdf) == {"x": [], "y": ["yintercept"]}
    # geom_function has no pre-stat data columns -> contributes nothing
    assert GeomFunction().position_aes_for_training(pd.DataFrame(index=[0])) == {"x": [], "y": []}


# -- geom_point --------------------------------------------------------------------


def test_point_centres_equal_transform_log_x():
    x = np.array([3.0, 30.0, 300.0, 2500.0])
    y = np.array([-1.0, 2.0, 0.5, 7.0])
    sx, sy, panel = _panel(x, y, scale_x_log10)  # trains in data space
    data = pd.DataFrame({"x": sx.map(x), "y": sy.map(y)})
    dots = GeomPoint().draw_panel(data, panel.panel_params, panel, theme_default())
    assert len(dots) == 4
    centres = np.array([d.get_center() for d in dots])
    np.testing.assert_allclose(centres, panel.transform(np.log10(x), y), atol=1e-9)


def test_point_mapped_colours_and_constant_override():
    cs = ScaleDiscrete("color", palette=[RED, BLUE])
    cs.train(pd.Series(["a", "b", "a"]))
    data = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0], "color": cs.map(pd.Series(["a", "b", "a"]))}
    )
    _, _, panel = _panel([0, 2], [0, 2])
    dots = GeomPoint().draw_panel(data, panel.panel_params, panel)
    cols = [d.get_fill_color().to_hex().upper() for d in dots]
    assert cols == [RED.to_hex().upper(), BLUE.to_hex().upper(), RED.to_hex().upper()]
    dots = GeomPoint().draw_panel(
        data, panel.panel_params, panel, color="#00FF00", size=0.2, alpha=0.5
    )
    assert {d.get_fill_color().to_hex().upper() for d in dots} == {"#00FF00"}
    assert all(np.isclose(d.width, 0.4) for d in dots)
    assert all(np.isclose(d.get_fill_opacity(), 0.5) for d in dots)


def test_point_drops_censored_rows_and_default_foreground():
    _, _, panel = _panel([0, 1], [0, 1])
    data = pd.DataFrame({"x": [0.0, np.nan, 1.0], "y": [0.0, 1.0, 1.0]})
    dots = GeomPoint().draw_panel(data, panel.panel_params, panel, theme_default())
    assert len(dots) == 2
    fg = theme_default().get_foreground().to_hex().upper()
    assert dots[0].get_fill_color().to_hex().upper() == fg


# -- geom_line ---------------------------------------------------------------------


def test_line_vertices_sorted_by_x_per_group():
    data = pd.DataFrame(
        {
            "x": [3.0, 1.0, 2.0, 0.0, 5.0, 4.0, 1.0],
            "y": [1.0, 2.0, 0.0, 3.0, 1.0, 2.0, 0.5],
            "group": [1, 1, 1, 1, 2, 2, 2],
            "color": [RED] * 4 + [BLUE] * 3,
        }
    )
    _, _, panel = _panel(data.x, data.y)
    out = GeomLine().draw_panel(data, panel.panel_params, panel)
    assert len(out) == 2 and len(out.marks) == 2 and len(out.labels) == 0
    for path, (_, g) in zip(out.marks, data.groupby("group")):
        g = g.sort_values("x")
        v = _vertices(path)
        assert len(v) == len(g)
        np.testing.assert_allclose(v, panel.transform(g.x, g.y), atol=1e-9)
    assert _color(out.marks[0]) == RED.to_hex().upper()
    assert _color(out.marks[1]) == BLUE.to_hex().upper()


def test_line_label_at_each_group_end_in_group_colour():
    data = pd.DataFrame(
        {
            "x": [0.0, 1.0, 0.0, 2.0],
            "y": [0.0, 1.0, 2.0, 3.0],
            "group": [1, 1, 2, 2],
            "color": [RED, RED, BLUE, BLUE],
        }
    )
    _, _, panel = _panel(data.x, data.y)
    out = GeomLine().draw_panel(data, panel.panel_params, panel, label=["a", "b"])
    assert len(out.labels) == 2 and len(out) == 4
    ends = panel.transform([1.0, 2.0], [1.0, 3.0])
    for lab, end, col in zip(out.labels, ends, (RED, BLUE)):
        assert lab.get_left()[0] > end[0]
        assert abs(lab.get_center()[1] - end[1]) < 1e-6
        assert lab.get_fill_color().to_hex().upper() == col.to_hex().upper()


def test_line_constant_colour_overrides_mapped():
    data = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "color": [RED, RED]})
    _, _, panel = _panel(data.x, data.y)
    out = GeomLine().draw_panel(data, panel.panel_params, panel, color=BLUE, linewidth=7)
    assert _color(out[0]) == BLUE.to_hex().upper()
    assert out[0].get_stroke_width() == 7


# -- geom_col ----------------------------------------------------------------------


def _col_panel(x_data, y, xs):
    sx, sy = xs(), scale_y_continuous()
    data = add_group(pd.DataFrame({"x": sx.transform(x_data), "y": np.asarray(y, float)}))
    data = GeomCol().setup_data(data, {})
    for ax, cols in GeomCol().position_aes_for_training(data).items():
        for c in cols:
            (sx if ax == "x" else sy).train_transformed(data[c])
    panel = CoordCartesian().build(sx, sy, theme_default())
    return data, panel


@pytest.mark.parametrize(
    "xs,x", [(scale_x_continuous, [1.0, 2.0, 4.0]), (scale_x_log10, [1.0, 10.0, 100.0])]
)
def test_col_rectangles_width_height(xs, x):
    y = [2.0, 5.0, 3.0]
    data, panel = _col_panel(x, y, xs)
    bars = GeomCol().draw_panel(data, panel.panel_params, panel)
    assert len(bars) == 3 and all(type(b) is VMobject for b in bars)
    (x0, x1), (y0, y1) = panel.panel_params.x_range, panel.panel_params.y_range
    ux = panel.axes.x_length / (x1 - x0)
    uy = panel.axes.y_length / (y1 - y0)
    xt = data["x"].to_numpy()
    res = resolution(xt)
    assert res == pytest.approx(1.0)  # 1,2,4 and log10 1,10,100 both have min gap 1
    for bar, xv, yv in zip(bars, xt, y):
        assert bar.width == pytest.approx(0.9 * res * ux)
        assert bar.height == pytest.approx(yv * uy)
        np.testing.assert_allclose(bar.get_bottom(), panel.transform(xv, 0.0)[0], atol=1e-9)
    # training on ymin includes 0: bars start inside the panel
    assert y0 < 0.0


def test_col_width_param_negative_values_and_clipping():
    sx, sy = scale_x_continuous(), scale_y_continuous(limits=(1, 6))
    data = pd.DataFrame({"x": [0.0, 1.0], "y": [4.0, -2.0], "group": -1})
    data = GeomCol().setup_data(data, {"width": 0.5})
    assert list(data.ymin) == [0.0, -2.0] and list(data.ymax) == [4.0, 0.0]
    sx.train_transformed(data.xmin)
    sx.train_transformed(data.xmax)
    panel = CoordCartesian().build(sx, sy, theme_default())
    bars = GeomCol().draw_panel(data, panel.panel_params, panel, fill=RED, alpha=0.5)
    lo = panel.transform(0.0, panel.panel_params.y_range[0])[0][1]
    assert bars[0].get_bottom()[1] == pytest.approx(lo)  # 0 is below the limits -> clipped
    assert bars[0].get_fill_color().to_hex().upper() == RED.to_hex().upper()
    assert bars[0].get_fill_opacity() == pytest.approx(0.5)
    assert bars[0].get_stroke_width() == 0


def test_col_draws_without_setup_data():
    _, _, panel = _panel([0, 3], [0, 3])
    bars = GeomCol().draw_panel(
        pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]}), panel.panel_params, panel
    )
    assert len(bars) == 2


# -- geom_vline / geom_hline --------------------------------------------------------


def test_vline_hline_span_panel_and_constant_colour():
    _, _, panel = _panel([0, 10], [-5, 5])
    pp = panel.panel_params
    v = GeomVline().draw_panel(
        pd.DataFrame({"xintercept": [2.0, 7.0]}), pp, panel, color=RED, label="v"
    )
    assert len(v.marks) == 2 and len(v.labels) == 2
    for line, xi in zip(v.marks, (2.0, 7.0)):
        np.testing.assert_allclose(line.get_start(), panel.transform(xi, pp.y_range[0])[0])
        np.testing.assert_allclose(line.get_end(), panel.transform(xi, pp.y_range[1])[0])
        assert _color(line) == RED.to_hex().upper()
    assert v.labels[0].get_fill_color().to_hex().upper() == RED.to_hex().upper()
    h = GeomHline().draw_panel(pd.DataFrame({"yintercept": [1.0]}), pp, panel)
    np.testing.assert_allclose(h[0].get_start(), panel.transform(pp.x_range[0], 1.0)[0])
    np.testing.assert_allclose(h[0].get_end(), panel.transform(pp.x_range[1], 1.0)[0])
    assert len(h) == 1


# -- geom_function / geom_smooth -----------------------------------------------------


def test_function_one_vmobject_per_group_with_label():
    sx, sy, _ = _panel([0, 2], [0, 4])
    df = GeomFunction().default_stat  # "function"
    assert df == "function"
    layer = geom_function(lambda x: x**2, color=RED, label="f")
    out = layer.stat.compute_layer(
        add_group(pd.DataFrame(index=[0])), {"x": sx, "y": sy}, **layer.stat.params
    )
    panel = CoordCartesian().build(sx, sy, theme_default())
    mob = layer.geom.draw_panel(out, panel.panel_params, panel, **layer.params)
    assert len(mob.marks) == 1 and len(mob.labels) == 1
    assert len(_vertices(mob.marks[0])) == 101
    assert _color(mob.marks[0]) == RED.to_hex().upper()
    np.testing.assert_allclose(_vertices(mob.marks[0])[-1], panel.transform(2.0, 4.0)[0], atol=1e-9)


@pytest.mark.parametrize("se", [True, False])
def test_smooth_one_mobject_per_group(se):
    rng = np.random.default_rng(1)
    x = np.tile(np.linspace(0, 10, 20), 2)
    g = np.repeat(["a", "b"], 20)
    y = np.where(g == "a", 1 + 2 * x, 5 - x) + rng.normal(0, 0.5, 40)
    df = add_group(pd.DataFrame({"x": x, "y": y, "color": g}))
    layer = geom_smooth(se=se)
    out = layer.stat.compute_layer(df, None, **layer.stat.params)
    cs = ScaleDiscrete("color", palette=[RED, BLUE])
    cs.train(out["color"])
    out["color"] = cs.map(out["color"])
    sx, sy = scale_x_continuous(), scale_y_continuous()
    for ax, cols in layer.geom.position_aes_for_training(out).items():
        for c in cols:
            (sx if ax == "x" else sy).train_transformed(out[c])
    panel = CoordCartesian().build(sx, sy, theme_default())
    mob = layer.geom.draw_panel(out, panel.panel_params, panel, **layer.params)
    assert len(mob) == 2
    for m, col in zip(mob, (RED, BLUE)):
        if se:
            assert isinstance(m, VGroup) and len(m) == 2
            ribbon, line = m
            assert ribbon.get_fill_opacity() == pytest.approx(0.4)
            assert ribbon.get_fill_color().to_hex().upper() == "#999999"
        else:
            assert not isinstance(m, VGroup)
            line = m
        assert _color(line) == col.to_hex().upper()
        assert len(_vertices(line)) == 80


# -- legend keys --------------------------------------------------------------------


@pytest.mark.parametrize(
    "geom",
    [GeomPoint(), GeomLine(), GeomCol(), GeomVline(), GeomHline(), GeomFunction(), GeomSmooth()],
)
def test_draw_key(geom):
    key = geom.draw_key({"color": RED}, theme_default())
    assert isinstance(key, VMobject)
    assert max(key.width, key.height) <= KEY_SIZE + 1e-9
    np.testing.assert_allclose(key.get_center(), 0, atol=1e-9)
    hexes = {
        m.get_stroke_color().to_hex().upper()
        for m in key.family_members_with_points()
        if m.get_stroke_width() > 0
    } | {
        m.get_fill_color().to_hex().upper()
        for m in key.family_members_with_points()
        if m.get_fill_opacity() > 0
    }
    if isinstance(geom, GeomCol):
        assert "#595959" in hexes  # default grey fill; color is the outline
    else:
        assert RED.to_hex().upper() in hexes


def test_vline_key_is_vertical():
    assert GeomVline().draw_key({}).height == pytest.approx(KEY_SIZE)
    assert GeomHline().draw_key({}).width == pytest.approx(KEY_SIZE)


# -- mixed layers with auto limits (geom level; the full pipeline is ggm-s11.8) ------


def _mini_build(plot):
    """A minimal stand-in for the build pipeline using the geom training API."""
    sx, sy = scale_x_continuous(), scale_y_continuous()
    scale = {"x": sx, "y": sy}
    frames = []
    for layer in plot.layers:
        df = evaluate_aes(layer.resolve_mapping(plot.mapping), layer.resolve_data(plot.data))
        df = add_group(df)
        for ax, cols in layer.geom.position_aes_for_training(df).items():
            for c in cols:
                scale[ax].train(df[c])
        frames.append(df)
    out = []
    for layer, df in zip(plot.layers, frames):
        df = df.copy()
        for c in df.columns:
            if c in X_AESTHETICS:
                df[c] = sx.transform(df[c])
            elif c in Y_AESTHETICS:
                df[c] = sy.transform(df[c])
        df = layer.stat.compute_layer(df, scale, **layer.stat.params)
        out.append(layer.geom.setup_data(df, layer.params))
    sx.reset()
    sy.reset()
    for layer, df in zip(plot.layers, out):
        for ax, cols in layer.geom.position_aes_for_training(df).items():
            for c in cols:
                scale[ax].train_transformed(df[c])
    panel = CoordCartesian().build(sx, sy, theme_default())
    mobs = [
        layer.geom.draw_panel(df, panel.panel_params, panel, **layer.params)
        for layer, df in zip(plot.layers, out)
    ]
    return panel, mobs


def test_mixed_function_vline_and_data_layers_auto_limits():
    df = pd.DataFrame({"a": [0.0, 2.0, 4.0], "b": [1.0, 3.0, 2.0]})
    p = (
        ggplot(df, aes("a", "b"))
        .geom_point()
        .geom_function(lambda x: x**2)
        .geom_vline(xintercept=6)
        .geom_hline(yintercept=-1)
    )
    panel, mobs = _mini_build(p)
    (x0, x1), (y0, y1) = panel.panel_params.x_range, panel.panel_params.y_range
    assert x0 < 0 and x1 > 6  # data + vline intercept
    assert y0 < -1 and y1 > 36  # hline + function evaluated over x in [0, 6]
    assert [len(m) for m in mobs] == [3, 1, 1, 1]
    # the function spans the x range trained by the other layers
    v = _vertices(mobs[1][0])
    np.testing.assert_allclose(v[[0, -1]], panel.transform([0.0, 6.0], [0.0, 36.0]), atol=1e-9)


def test_geom_base_draw_panel_requires_aes():
    _, _, panel = _panel([0, 1], [0, 1])
    with pytest.raises(ValueError, match="missing aesthetics: y"):
        GeomPoint().draw_panel(pd.DataFrame({"x": [1.0]}), panel.panel_params, panel)
    assert isinstance(GeomPoint(), Geom)


def test_groups_inferred_from_mapped_colours_when_group_missing():
    data = pd.DataFrame(
        {"x": [0.0, 1.0, 0.0, 1.0], "y": [0.0, 1.0, 1.0, 0.0], "color": [RED, RED, BLUE, BLUE]}
    )
    _, _, panel = _panel(data.x, data.y)
    out = GeomLine().draw_panel(data, panel.panel_params, panel)
    assert len(out) == 2
