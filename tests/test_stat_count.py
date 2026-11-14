"""stat_count / geom_bar (counting)."""

import numpy as np
import pandas as pd
import pytest

from ggmanim import (
    aes,
    after_stat,
    geom_bar,
    geom_col,
    ggplot,
    position_dodge,
    scale_x_log10,
    stat_count,
)
from ggmanim.geoms import GeomBar, GeomCol
from ggmanim.stats import STATS, StatCount, StatIdentity

DF = pd.DataFrame({"cut": list("aabbbc"), "g": list("xyxyxy"), "w": [1, 2, 3, 4, 5, 6.0]})


def test_registered_and_defaults():
    assert STATS["count"] is StatCount
    layer = geom_bar()
    assert isinstance(layer.geom, GeomBar) and isinstance(layer.geom, GeomCol)
    assert isinstance(layer.stat, StatCount)
    assert layer.mapping["y"] == after_stat("count")
    assert isinstance(geom_col().stat, StatIdentity)
    assert isinstance(geom_bar(stat="identity").stat, StatIdentity)
    assert "y" not in geom_bar(stat="identity").mapping


def test_compute_counts_props_and_weight():
    df = pd.DataFrame({"x": [1.0, 1.0, 2.0, 3.0, 3.0, 3.0]})
    out = StatCount().compute(df)
    assert out["x"].tolist() == [1.0, 2.0, 3.0]
    assert out["count"].tolist() == [2, 1, 3]
    np.testing.assert_allclose(out["prop"], [2 / 6, 1 / 6, 3 / 6])
    np.testing.assert_allclose(out["width"], 0.9)
    assert (out["y"] == out["count"]).all() and not out["flipped_aes"].any()
    out = StatCount().compute(df.assign(weight=[1, 1, 0.5, 2, 2, np.nan]))
    assert out["count"].tolist() == [2, 0.5, 4]
    assert "weight" not in out.columns


def test_geom_bar_counts_a_string_column():
    m = ggplot(DF, aes("cut")).add(geom_bar()).build()
    d = m.layer_data[0]
    assert d["count"].tolist() == [2, 3, 1]
    assert d["y"].tolist() == [2, 3, 1]
    np.testing.assert_allclose(d["xmax"] - d["xmin"], 0.9)
    assert len(m.layers[0].marks) == 3
    assert m.panel.y_title.text == "count"


def test_weight_and_prop_mapping():
    m = ggplot(DF, aes("cut", weight="w")).add(geom_bar()).build()
    assert m.layer_data[0]["y"].tolist() == [3, 12, 6]
    m = ggplot(DF, aes("cut")).add(geom_bar(aes(y=after_stat("prop"), group=1))).build()
    np.testing.assert_allclose(m.layer_data[0]["y"], [2 / 6, 3 / 6, 1 / 6])
    assert m.panel.y_title.text == "prop"


def test_stacked_and_dodged_counts():
    m = ggplot(DF, aes("cut", fill="g")).add(geom_bar()).build()
    d = m.layer_data[0]
    top = d.groupby("x")["ymax"].max()
    assert top.tolist() == [2, 3, 1]
    m = ggplot(DF, aes("cut", fill="g")).add(geom_bar(position=position_dodge())).build()
    d = m.layer_data[0]
    assert (d["ymin"] == 0).all() and len(d) == 5
    a = d[d["x"].between(0.5, 1.5)]
    assert a["xmax"].min() <= a["xmin"].max() + 1e-9  # side by side


def test_width_param_and_continuous_x():
    df = pd.DataFrame({"x": [2.0, 2.0, 4.0, 6.0]})
    out = ggplot(df, aes("x")).add(geom_bar(width=0.5)).build().layer_data[0]
    np.testing.assert_allclose(out["xmax"] - out["xmin"], 0.5)
    out = ggplot(df, aes("x")).add(geom_bar()).build().layer_data[0]
    np.testing.assert_allclose(out["xmax"] - out["xmin"], 1.8)  # 0.9 * resolution 2


def test_log_x_counts_in_transformed_space():
    df = pd.DataFrame({"x": [1.0, 10.0, 10.0, 100.0]})
    d = ggplot(df, aes("x")).add(geom_bar()).add(scale_x_log10()).build().layer_data[0]
    np.testing.assert_allclose(d["x"], [0, 1, 2])
    assert d["count"].tolist() == [1, 2, 1]


def test_stat_count_layer_and_y_rejected():
    m = ggplot(DF, aes("cut")).add(stat_count()).build()
    assert isinstance(m.layer_data[0], pd.DataFrame) and len(m.layers[0].marks) == 3
    with pytest.raises(Exception, match="only have an x"):
        StatCount().compute(pd.DataFrame({"x": [1.0], "y": [2.0]}))
