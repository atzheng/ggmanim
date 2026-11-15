"""Stat-defaults cleanup (ggm-ydy): make_layer alone adds stat defaults, the
once-per-layer ``Stat.setup_layer`` hook, horizontal missing-value warnings and
``geom_col`` orientation."""

import warnings

import numpy as np
import pandas as pd
import pytest

from ggmanim import (
    aes,
    facet_wrap,
    geom_bar,
    geom_col,
    geom_histogram,
    ggplot,
    stat_bin,
    stat_count,
    stat_sum,
)
from ggmanim.geoms.base import StatDefault
from ggmanim.stats import Stat, StatBin, StatCount


def _data(plot):
    return plot.build().layer_data[0]


def test_stat_layer_defaults_come_from_make_layer():
    for layer in (stat_count(), stat_sum(), stat_bin(binwidth=1)):
        assert any(isinstance(v, StatDefault) for v in layer.mapping.values())
    # a user's own after_stat default is kept as given (not a StatDefault)
    from ggmanim import after_stat

    lay = stat_count(aes(y=after_stat("count")))
    assert not isinstance(lay.mapping["y"], StatDefault)


def test_setup_layer_hook_is_a_noop_by_default_and_runs_once():
    calls = []

    class Spy(StatCount):
        def setup_layer(self, df, params):
            calls.append(len(df))

    df = pd.DataFrame({"x": list("abcabc"), "f": list("pqrpqr")})
    ggplot(df, aes("x")).add(geom_bar(stat=Spy())).add(facet_wrap("f")).build()
    assert calls == [6]
    assert Stat.setup_layer(StatCount(), df, {}) is None


def test_bins_warning_once_per_layer_under_facets():
    df = pd.DataFrame({"v": np.arange(12.0), "f": list("abc") * 4})
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ggplot(df, aes("v")).add(geom_histogram()).add(facet_wrap("f")).build()
    assert sum("bins = 30" in str(w.message) for w in rec) == 1
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        StatBin().compute(df[["v"]].rename(columns={"v": "x"}))
    assert sum("bins = 30" in str(w.message) for w in rec) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ggplot(df, aes("v")).add(geom_histogram(binwidth=2)).add(facet_wrap("f")).build()


@pytest.mark.parametrize("geom", [geom_bar, geom_histogram])
def test_horizontal_layers_warn_about_missing_values(geom):
    vals = ["a", None, "b", "a"] if geom is geom_bar else [1.0, np.nan, 3.0, 4.0]
    df = pd.DataFrame({"v": vals})
    for m in (aes("v"), aes(y="v")):
        with pytest.warns(UserWarning, match="removed 1 row"):
            ggplot(df, m).add(geom(**({} if geom is geom_bar else {"bins": 3}))).build()


def test_geom_col_horizontal_from_discrete_y():
    df = pd.DataFrame({"k": list("abc"), "v": [1.0, 2.0, 3.0]})
    d = _data(ggplot(df, aes("v", "k")).add(geom_col()))
    assert d["flipped_aes"].all()
    np.testing.assert_allclose(d["xmin"], 0)
    np.testing.assert_allclose(d["xmax"], [1, 2, 3])
    np.testing.assert_allclose(d["ymin"], [0.55, 1.55, 2.55])
    np.testing.assert_allclose(d["ymax"], [1.45, 2.45, 3.45])


def test_geom_col_vertical_unchanged_and_orientation_param():
    df = pd.DataFrame({"k": list("abc"), "v": [1.0, 2.0, 3.0]})
    d = _data(ggplot(df, aes("k", "v")).add(geom_col()))
    assert "flipped_aes" not in d.columns or not d["flipped_aes"].any()
    np.testing.assert_allclose(d["ymax"], [1, 2, 3])
    num = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [4.0, 5.0, 6.0]})
    h = _data(ggplot(num, aes("x", "y")).add(geom_col(orientation="y", width=0.5)))
    np.testing.assert_allclose(h["xmax"], [4, 5, 6])
    np.testing.assert_allclose(h["ymax"] - h["ymin"], 0.5)
    with pytest.raises(ValueError, match="orientation"):
        ggplot(num, aes("x", "y")).add(geom_col(orientation="z")).build()


def test_geom_col_horizontal_stacks_along_x():
    df = pd.DataFrame({"k": list("aabb"), "g": list("xyxy"), "v": [1.0, 2.0, 3.0, 4.0]})
    d = _data(ggplot(df, aes("v", "k", fill="g")).add(geom_col()))
    assert d["flipped_aes"].all()
    assert sorted(d["xmax"].tolist()) == [2.0, 3.0, 4.0, 7.0]
