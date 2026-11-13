"""Discrete position scales (scale_x_discrete / scale_y_discrete)."""

import warnings

import numpy as np
import pandas as pd
import pytest

import ggmanim
from ggmanim import aes, ggplot, scale_x_discrete, scale_y_discrete
from ggmanim.build import BuildError
from ggmanim.scales import ScaleContinuousPosition, ScaleDiscretePosition
from ggmanim.scales.continuous import X_AESTHETICS, Y_AESTHETICS


# ----------------------------------------------------------------- scale class (no build)
def test_constructors_and_aesthetics():
    sx, sy = scale_x_discrete(), scale_y_discrete()
    assert isinstance(sx, ScaleDiscretePosition) and sx.is_discrete
    assert sx.aesthetics == X_AESTHETICS and sy.aesthetics == Y_AESTHETICS
    assert scale_x_discrete(aesthetics="x").aesthetics == ("x",)
    assert "scale_x_discrete" in ggmanim.__all__ and "scale_y_discrete" in ggmanim.__all__


def test_strings_sorted_and_mapped_to_positions():
    s = scale_x_discrete()
    s.train(pd.Series(["b", "a", "c", "a", None]))
    assert s.get_limits() == ["a", "b", "c"]
    np.testing.assert_array_equal(s.transform(["c", "a", "b"]), [3.0, 1.0, 2.0])
    assert np.isnan(s.transform([None])[0])
    np.testing.assert_array_equal(s.get_breaks(), [1.0, 2.0, 3.0])
    assert s.get_labels() == ["a", "b", "c"]


def test_categorical_order_and_drop():
    cat = pd.Categorical(["hi", "lo"], categories=["lo", "mid", "hi"])
    s = scale_x_discrete()
    s.train(pd.Series(cat))
    assert s.get_limits() == ["lo", "hi"]  # unused "mid" dropped
    s = scale_x_discrete(drop=False)
    s.train(cat)
    assert s.get_limits() == ["lo", "mid", "hi"]


def test_categoricals_accumulate_in_encounter_order():
    s = scale_x_discrete()
    s.train(pd.Categorical(["z", "y"], categories=["z", "y"]))
    s.train(pd.Categorical(["a", "y"], categories=["y", "a"]))
    assert s.get_limits() == ["z", "y", "a"]


def test_numeric_values_pass_through_and_train_continuous_range():
    s = scale_x_discrete()
    s.train(["a", "b"])
    s.train([0.5, 2.5])
    assert s.get_limits() == ["a", "b"]
    np.testing.assert_array_equal(s.transform([1.5]), [1.5])
    assert s.range == (0.5, 2.5)
    # (1, 2) expanded by 0.6 is (0.4, 2.6), which already covers (0.5, 2.5)
    assert s.get_expanded_limits() == pytest.approx((0.4, 2.6))
    s.train_transformed([-1.0])
    assert s.get_expanded_limits() == pytest.approx((-1.0, 2.6))


def test_expansion_default_and_override():
    s = scale_x_discrete()
    s.train(["a", "b", "c"])
    assert s.dimension() == pytest.approx((0.4, 3.6))
    assert s.get_expanded_limits(False) == (1.0, 3.0)
    s2 = scale_x_discrete(expand=(0, 0))
    s2.train(["a", "b", "c"])
    assert s2.get_expanded_limits() == (1.0, 3.0)
    s3 = scale_x_discrete(expand=(0, 1.0))
    s3.train(["a", "b"])
    assert s3.get_expanded_limits() == (0.0, 3.0)


def test_empty_scale():
    s = scale_x_discrete()
    assert s.get_limits() == [] and s.get_expanded_limits() is None and s.is_empty()
    assert len(s.get_breaks()) == 0 and s.get_labels() == []


def test_reset_keeps_levels_clone_clears_them():
    s = scale_x_discrete()
    s.train(["a", "b"])
    s.train_transformed([0.0, 5.0])
    s.reset()
    assert s.get_limits() == ["a", "b"] and s.range is None
    c = s.clone()
    assert c.get_limits() == [] and s.get_limits() == ["a", "b"]


def test_limits_reorder_subset_and_warn():
    s = scale_x_discrete(limits=["c", "a"])
    s.train(["a", "b", "c"])
    assert s.get_limits() == ["c", "a"]
    with pytest.warns(UserWarning, match="Removed 1 value"):
        out = s.transform(["a", "b", "c"])
    np.testing.assert_array_equal(out, [2.0, np.nan, 1.0])


def test_breaks_and_labels_overrides():
    s = scale_x_discrete(breaks=["c", "a", "zz"], labels=["C", "A", "Z"])
    s.train(["a", "b", "c"])
    np.testing.assert_array_equal(s.get_breaks(), [3.0, 1.0])
    assert s.get_labels() == ["C", "A"]
    s = scale_x_discrete(labels={"a": "Alpha"})
    s.train(["a", "b"])
    assert s.get_labels() == ["Alpha", "b"]
    s = scale_x_discrete(labels=str.upper)
    s.train(["a", "b"])
    assert s.get_labels() == ["A", "B"]
    s = scale_x_discrete(labels=["one", "two"])
    s.train(["a", "b"])
    assert s.get_labels() == ["one", "two"]
    s = scale_x_discrete(breaks=[])
    s.train(["a", "b"])
    assert len(s.get_breaks()) == 0
    with pytest.raises(ValueError, match="same length"):
        scale_x_discrete(breaks=["a"], labels=["x", "y"])


def test_breaks_censored_to_numeric_limits():
    s = scale_x_discrete()
    s.train(["a", "b", "c"])
    np.testing.assert_array_equal(s.get_breaks((1.5, 3.6)), [2.0, 3.0])
    assert s.get_labels((1.5, 3.6)) == ["b", "c"]


# ----------------------------------------------------------------- build
@pytest.fixture
def bars():
    return pd.DataFrame({"category": ["b", "a", "c"], "value": [3.0, 1.0, 2.0]})


def test_geom_col_with_string_x(bars):
    m = ggplot(bars, aes("category", "value")).geom_col().build()
    assert isinstance(m.scales["x"], ScaleDiscretePosition)
    assert isinstance(m.scales["y"], ScaleContinuousPosition)
    pp = m.panel_params
    assert pp.x_breaks == (1.0, 2.0, 3.0) and pp.x_labels == ("a", "b", "c")
    assert pp.x_range == pytest.approx((0.4, 3.6))
    d = m.layer_data[0]
    np.testing.assert_array_equal(d["x"], [2.0, 1.0, 3.0])
    np.testing.assert_allclose(d["xmax"] - d["xmin"], 0.9)
    # tick labels are the level names; axis title is the column name
    assert [t.text for t in m.panel.x_tick_labels] == ["a", "b", "c"]
    assert m.panel.x_title.text == "category"


def test_geom_col_width_on_discrete_axis_ignores_gaps(bars):
    # only levels 1 and 3 present after subsetting: width stays 0.9, not 0.9 * 2
    m = ggplot(bars, aes("category", "value")).geom_col().scale_x_discrete(limits=["a", "x", "c"])
    with pytest.warns(UserWarning):
        m = m.build()
    d = m.layer_data[0].dropna(subset=["x"])
    np.testing.assert_allclose(d["xmax"] - d["xmin"], 0.9)


def test_points_with_categorical_x_in_explicit_order():
    df = pd.DataFrame(
        {
            "g": pd.Categorical(["hi", "lo", "mid"], categories=["lo", "mid", "hi"]),
            "v": [1.0, 2.0, 3.0],
        }
    )
    m = ggplot(df, aes("g", "v")).geom_point().build()
    assert m.panel_params.x_labels == ("lo", "mid", "hi")
    np.testing.assert_array_equal(m.layer_data[0]["x"], [3.0, 1.0, 2.0])
    centres = np.array([d.get_center() for d in m.layers[0]])
    xs = m.panel.transform([3.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(centres[:, :2], xs[:, :2], atol=1e-6)


def test_discrete_y(bars):
    m = ggplot(bars, aes("value", "category")).geom_point().build()
    assert m.panel_params.y_labels == ("a", "b", "c")
    assert isinstance(m.scales["y"], ScaleDiscretePosition)


def test_limits_subsetting_drops_rows_and_warns(bars):
    p = ggplot(bars, aes("category", "value")).geom_point().scale_x_discrete(limits=["c", "a"])
    with pytest.warns(UserWarning, match="outside the limits"):
        m = p.build()
    assert m.panel_params.x_labels == ("c", "a")
    assert np.isnan(m.layer_data[0]["x"][0])  # "b" removed
    assert len(m.layers[0]) == 2


def test_labels_override_and_name(bars):
    p = ggplot(bars, aes("category", "value")).geom_col()
    m = p.add(scale_x_discrete(name="Kind", labels={"a": "Apple"})).build()
    assert m.panel_params.x_labels == ("Apple", "b", "c")
    assert m.panel.x_title.text == "Kind"


def test_expand_override_in_build(bars):
    m = ggplot(bars, aes("category", "value")).geom_point().scale_x_discrete(expand=(0, 0)).build()
    assert m.panel_params.x_range == (1.0, 3.0)


def test_mixed_layers_accumulate_levels():
    a = pd.DataFrame({"k": pd.Categorical(["q", "p"], categories=["q", "p"]), "v": [1.0, 2.0]})
    b = pd.DataFrame({"k": pd.Categorical(["r", "p"], categories=["r", "p"]), "v": [3.0, 4.0]})
    m = ggplot(mapping=aes("k", "v")).geom_point(data=a).geom_line(data=b).build()
    assert m.panel_params.x_labels == ("q", "p", "r")
    np.testing.assert_array_equal(m.layer_data[1]["x"], [3.0, 2.0])


def test_numeric_layer_on_discrete_axis(bars):
    m = (
        ggplot(bars, aes("category", "value"))
        .geom_col()
        .geom_vline(xintercept=1.5)
        .geom_point(data=pd.DataFrame({"category": [2.5], "value": [1.0]}))
        .build()
    )
    assert m.panel_params.x_labels == ("a", "b", "c")
    np.testing.assert_array_equal(m.layer_data[2]["x"], [2.5])


def test_user_continuous_scale_with_strings_errors(bars):
    with pytest.raises(BuildError, match="scale_x_discrete"):
        ggplot(bars, aes("category", "value")).geom_point().scale_x_continuous().build()


def test_coord_zoom_on_discrete_axis(bars):
    m = ggplot(bars, aes("category", "value")).geom_point().coord_cartesian(xlim=(1, None)).build()
    lo, hi = m.panel_params.x_range
    assert lo < 1.0 < 3.0 < hi


def test_build_is_repeatable_and_spec_scale_untouched(bars):
    sx = scale_x_discrete()
    p = ggplot(bars, aes("category", "value")).geom_col().add(sx)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m1, m2 = p.build(), p.build()
    assert m1.panel_params == m2.panel_params
    assert sx.get_limits() == []
