"""Structural tests for the build pipeline (spec -> GGPlot)."""

import numpy as np
import pandas as pd
import pytest
from manim import RED, Dot, Square, VGroup, VMobject

import ggmanim.coords  # noqa: F401  (registers components)
import ggmanim.geoms  # noqa: F401
import ggmanim.scales  # noqa: F401
import ggmanim.scales.discrete  # noqa: F401
import ggmanim.stats  # noqa: F401
import ggmanim.theme  # noqa: F401
from ggmanim.aes import aes, after_stat
from ggmanim.build import BuildError, build_plot
from ggmanim.coords import CartesianPanel
from ggmanim.mobjects import GGPlot, LayerDict
from ggmanim.plot import GGManim, ggplot
from ggmanim.scales import scale_x_log10, xlim
from ggmanim.scales.discrete import scale_color_manual
from ggmanim.theme import theme


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [2.0, 4.0, 3.0, 5.0, 4.5, 6.0],
            "g": ["x", "y", "x", "y", "x", "y"],
        }
    )


def _hex(c) -> str:
    return c.to_hex().upper()


def _centers(group) -> np.ndarray:
    return np.array([m.get_center() for m in group])


def _c2p(m: GGPlot, x, y) -> np.ndarray:
    return m.panel.transform(x, y)


def _vertices(v: VMobject) -> np.ndarray:
    return np.vstack([v.get_start_anchors(), v.get_end_anchors()[-1:]])


# -- end-to-end per geom ----------------------------------------------------------


def test_point_with_colour_mapping(df):
    m = ggplot(df, aes("a", "b", color="g")).geom_point().build()
    assert isinstance(m, GGPlot) and isinstance(m.panel, CartesianPanel)
    dots = m.layers[0]
    assert len(dots) == 6 and all(isinstance(d, Dot) for d in dots)
    np.testing.assert_allclose(_centers(dots), _c2p(m, df["a"], df["b"]), atol=1e-6)
    cols = [_hex(d.get_fill_color()) for d in dots]
    assert cols[0] == cols[2] == cols[4] != cols[1] == cols[3] == cols[5]
    assert m.scales["color"].get_limits() == ["x", "y"]
    # axis titles default to the mapped column names
    assert m.panel.x_title.text == "a" and m.panel.y_title.text == "b"


def test_line_groups_by_colour(df):
    m = ggplot(df, aes("a", "b", color="g")).geom_line().build()
    paths = m.layers[0].marks if hasattr(m.layers[0], "marks") else m.layers[0]
    assert len(paths) == 2
    assert {_hex(p.get_stroke_color()) for p in paths} == {
        _hex(c) for c in m.scales["color"].get_palette_values()
    }
    xs = df[df.g == "x"]
    np.testing.assert_allclose(_vertices(paths[0]), _c2p(m, xs["a"], xs["b"]), atol=1e-6)


def test_col_with_fill_mapping_includes_zero(df):
    m = ggplot(df, aes("a", "b", fill="g")).geom_col().build()
    assert len(m.layers[0]) == 6
    lo, hi = m.scales["y"].get_limits()
    assert lo == 0.0 and hi == 6.0  # bars start at 0, so y trains on ymin = 0
    assert m.layer_data[0]["fill"].nunique() == 2


def test_vline_hline(df):
    m = (
        ggplot(df, aes("a", "b"))
        .geom_point()
        .geom_vline(xintercept=[2, 8], color=RED)
        .geom_hline(yintercept=-1)
        .build()
    )
    assert len(m.layers[1]) == 2 and len(m.layers[2]) == 1
    assert m.scales["x"].get_limits() == (1.0, 8.0)
    assert m.scales["y"].get_limits() == (-1.0, 6.0)
    vline = m.layers[1][1]
    assert _hex(vline.get_stroke_color()) == _hex(RED)
    x0 = _c2p(m, 8.0, 0.0)[0, 0]
    assert vline.get_start()[0] == pytest.approx(x0)


def test_function_uses_trained_x_range(df):
    m = ggplot(df, aes("a", "b")).geom_point().geom_function(lambda x: x**2, n=11).build()
    curve = m.layers[1][0]
    d = m.layer_data[1]
    assert len(d) == 11 and d["x"].min() == 1.0 and d["x"].max() == 6.0
    np.testing.assert_allclose(d["y"], d["x"] ** 2)
    assert m.scales["y"].get_limits() == (1.0, 36.0)  # the curve trains y
    np.testing.assert_allclose(_vertices(curve), _c2p(m, d["x"], d["y"]), atol=1e-6)


def test_function_alone_with_xlim():
    m = ggplot().geom_function(np.sin, xlim=(0, 3)).build()
    assert m.scales["x"].get_limits() == (0.0, 3.0)
    assert len(m.layers[0]) == 1


def test_smooth_per_colour_group(df):
    m = ggplot(df, aes("a", "b", color="g")).geom_point().geom_smooth().build()
    smooth = m.layers[1]
    assert len(smooth) == 2
    d = m.layer_data[1]
    assert set(d["group"]) == {1, 2} and {"ymin", "ymax"} <= set(d.columns)
    # the ribbon trains y beyond the data
    assert m.scales["y"].get_limits()[0] <= d["ymin"].min()


# -- purity / idempotence -----------------------------------------------------------


def test_idempotent_and_pure(df):
    df_before = df.copy()
    user_scale = scale_color_manual({"x": "red", "y": "blue"})
    title = Square(side_length=0.2)
    p = (
        ggplot(df, aes("a", "b", color="g"))
        .geom_point()
        .geom_smooth(se=False)
        .geom_vline(xintercept=3)
        .add(user_scale)
        .labs(x=title, title="T")
    )
    layers_before, scales_before = p.layers, p.scales
    m1, m2 = p.build(), p.build()
    assert m1.panel_params == m2.panel_params
    assert list(m1.layers) == list(m2.layers) == [0, 1, 2]
    for k in m1.layers:
        np.testing.assert_allclose(m1.layers[k].get_all_points(), m2.layers[k].get_all_points())
    np.testing.assert_allclose(m1.get_all_points(), m2.get_all_points())
    pd.testing.assert_frame_equal(df, df_before)
    assert p.layers is layers_before and p.scales is scales_before
    assert user_scale.get_limits() == []  # spec scale never trained
    assert m1.panel.x_title is not title
    np.testing.assert_allclose(title.get_center(), [0, 0, 0])
    assert m1.title.text == "T"


def test_legacy_chain(df):
    m = GGManim(df, aes(x="a", y="b", color="g")).geom_point().build()
    assert isinstance(m["layers"], LayerDict)
    assert len(m["layers"][0]) == 6
    assert m["axes"] is m.axes is m.panel.axes
    assert isinstance(m["legends"], VGroup)
    assert isinstance(m[0], CartesianPanel)  # int indexing still indexes submobjects


def test_named_layers(df):
    m = ggplot(df, aes("a", "b")).geom_point(name="pts").geom_line().build()
    assert list(m.layers) == ["pts", 1]
    assert m.layers["pts"] is m.layers[0]
    assert m.layers[1] is not m.layers["pts"]
    with pytest.raises(KeyError):
        m.layers["nope"]
    with pytest.raises(KeyError):
        m["nope"]


# -- position training -------------------------------------------------------------


def test_auto_limits_mixed_function_intercepts_and_data(df):
    # regression: the old builder did layer.data[layer.aes["x"]] for every layer -> KeyError
    m = (
        ggplot(df, aes("a", "b"))
        .geom_function(np.sqrt)
        .geom_vline(xintercept=10)
        .geom_hline(yintercept=-2)
        .geom_point()
        .build()
    )
    assert m.scales["x"].get_limits() == (1.0, 10.0)
    lo, hi = m.scales["y"].get_limits()
    assert lo == -2.0 and hi == 6.0
    # the function spans the x range trained before the stat (data + intercept)
    d = m.layer_data[0]
    assert d["x"].min() == 1.0 and d["x"].max() == 10.0


def test_per_layer_data_override(df):
    other = pd.DataFrame({"a": [10.0, 20.0], "b": [-5.0, 0.0]})
    m = ggplot(df, aes("a", "b")).geom_point().geom_line(data=other).build()
    assert len(m.layers[0]) == 6
    assert m.layer_data[1]["x"].tolist() == [10.0, 20.0]
    assert m.scales["x"].get_limits() == (1.0, 20.0)
    assert m.scales["y"].get_limits() == (-5.0, 6.0)


def test_layer_data_callable_and_inherit_aes_false(df):
    m = (
        ggplot(df, aes("a", "b", color="g"))
        .geom_point(data=lambda d: d[d.a > 3], mapping=aes("a", "b"), inherit_aes=False)
        .build()
    )
    assert len(m.layers[0]) == 3 and "color" not in m.layer_data[0]
    assert "color" not in m.scales


def test_log10_x_end_to_end():
    d = pd.DataFrame({"x": [1.0, 10.0, 100.0, 1000.0], "y": [1.0, 2.0, 3.0, 4.0]})
    m = ggplot(d, aes("x", "y")).geom_point().geom_vline(xintercept=10).add(scale_x_log10()).build()
    assert m.scales["x"].get_limits() == pytest.approx((0.0, 3.0))
    np.testing.assert_allclose(_centers(m.layers[0]), _c2p(m, [0, 1, 2, 3], d["y"]), atol=1e-6)
    labels = dict(zip(m.panel_params.x_labels, m.panel_params.x_breaks))
    assert {k: labels[k] for k in ("1", "10", "100", "1000")} == pytest.approx(
        {"1": 0.0, "10": 1.0, "100": 2.0, "1000": 3.0}
    )
    assert m.layers[1][0].get_start()[0] == pytest.approx(_c2p(m, 1.0, 0)[0, 0])


def test_user_limits_censor(df):
    m = ggplot(df, aes("a", "b")).geom_point().add(xlim(2, 4)).build()
    assert m.scales["x"].get_limits() == (2.0, 4.0)
    assert len(m.layers[0]) == 3  # out-of-bounds rows -> NaN -> not drawn


# -- labels / titles / theme ----------------------------------------------------------


def test_labs_override_and_title(df):
    m = (
        ggplot(df, aes("a", "b"))
        .geom_point()
        .labs(x="Dose", y=None, title="Title", subtitle="Sub", caption="Cap")
        .build()
    )
    assert m.panel.x_title.text == "Dose" and m.panel.y_title is None
    assert m["title"] is m.title and m.title.text == "Title"
    assert m.title.get_bottom()[1] >= m.subtitle.get_top()[1] - 1e-6
    assert m.subtitle.get_bottom()[1] >= m.panel.get_top()[1] - 1e-6
    assert m.caption.get_top()[1] <= m.panel.get_bottom()[1] + 1e-6


def test_after_stat_mapping(df):
    m = ggplot(df, aes("a", "b")).geom_line(aes(y=after_stat("x"))).build()
    assert len(m.layers[0]) == 1
    np.testing.assert_allclose(m.layer_data[0]["y"], df["a"])


def test_no_layers_builds_empty_panel():
    m = ggplot().build()
    assert len(m.layers) == 0 and m.panel_params.x_range == (0.0, 1.0)


def test_legend_none_theme(df):
    m = ggplot(df, aes("a", "b", color="g")).geom_point().add(theme(legend_position="none")).build()
    assert len(m.legend) == 0 and m.legends == []


# -- errors ------------------------------------------------------------------------


def test_unknown_column_names_column_and_layer(df):
    p = ggplot(df, aes("a", "b")).geom_point().geom_line(aes(color="nope"), name="lines")
    with pytest.raises(BuildError, match=r"layer 'lines' \(GeomLine\).*'color'.*'nope'"):
        p.build()
    p = ggplot(df, aes("a", "zzz")).geom_point()
    with pytest.raises(BuildError, match=r"layer 0 \(GeomPoint\).*'zzz'"):
        p.build()


def test_discrete_x_on_continuous_user_scale_errors(df):
    # a string column gets a discrete x scale by default (tests/test_scales_discrete_position.py),
    # but an explicit continuous scale cannot take it
    with pytest.raises(BuildError, match="scale_x_discrete"):
        ggplot(df, aes("g", "b")).geom_point().add(xlim(0, 1)).build()


def test_numeric_colour_builds_colourbar(df):
    from ggmanim.guides import Colourbar
    from ggmanim.scales.colour import ScaleContinuousColour

    m = ggplot(df, aes("a", "b", color="b")).geom_point().build()
    assert isinstance(m.scales["color"], ScaleContinuousColour)
    (cb,) = m.legends
    assert isinstance(cb, Colourbar)


def test_build_plot_theme_override(df):
    p = ggplot(df, aes("a", "b")).geom_point()
    m = build_plot(p, theme=theme(panel_width=4.0, panel_height=3.0))
    assert m.axes.x_length == pytest.approx(4.0)


# -- legends (guides module) --------------------------------------------------------


def test_legend_built_and_placed_right_of_panel(df):
    from ggmanim.guides import Legend

    m = ggplot(df, aes("a", "b", color="g")).geom_point().geom_line().build()
    assert len(m.legends) == 1 and isinstance(m.legends[0], Legend)
    assert m["legends"] is m.legend and len(m.legend) == 1
    assert m.legend.get_left()[0] > m.panel.get_right()[0]
    assert m.legends[0].breaks == ["x", "y"]
    # room was reserved for the legend: the panel is narrower than without one
    plain = ggplot(df, aes("a", "b")).geom_point().build()
    assert m.axes.x_length < plain.axes.x_length
    assert len(plain.legends) == 0


def test_guides_none_hides_legend(df):
    m = ggplot(df, aes("a", "b", color="g")).geom_point().guides(color="none").build()
    assert m.legends == [] and len(m.legend) == 0
