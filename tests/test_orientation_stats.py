"""Horizontal stat_count / stat_bin (orientation "y"), the stat's after_stat
defaults added by make_layer, and the once-per-build bins warning (ggm-d5h).

Expected values are ggplot2 4.0.3 ``layer_data()`` output (Rscript)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from ggmanim import (
    aes,
    after_stat,
    facet_wrap,
    geom_bar,
    geom_freqpoly,
    geom_histogram,
    geom_point,
    ggplot,
    stat_bin,
    stat_count,
)
from ggmanim.build import BuildError
from ggmanim.geoms.base import StatDefault
from ggmanim.stats import StatBin, StatCount

CUT = pd.DataFrame({"cut": list("aabbbc"), "g": list("xyxyxy"), "w": [1, 2, 3, 4, 5, 6.0]})
V = [0.5, 1.2, 2.5, 2.7, 3.1, 4.8, 5.0, 5.5, 7.9, 9.6]


def _data(plot):
    return plot.build().layer_data[0]


# -- stat_count -------------------------------------------------------------------


def test_bar_on_y_counts_per_y():
    # ggplot(d, aes(y = cut, weight = w)) + geom_bar()
    d = _data(ggplot(CUT, aes(y="cut", weight="w")).add(geom_bar()))
    assert d["count"].tolist() == [3, 12, 6]
    assert d["x"].tolist() == [3, 12, 6] and d["y"].tolist() == [1, 2, 3]
    np.testing.assert_allclose(d["xmin"], 0)
    np.testing.assert_allclose(d["xmax"], [3, 12, 6])
    np.testing.assert_allclose(d["ymin"], [0.55, 1.55, 2.55])
    np.testing.assert_allclose(d["ymax"], [1.45, 2.45, 3.45])
    assert d["flipped_aes"].all()


def test_bar_on_continuous_y_uses_y_resolution():
    # ggplot(data.frame(v = c(1, 1, 2, 4, 4, 4)), aes(y = v)) + geom_bar()
    df = pd.DataFrame({"v": [1.0, 1, 2, 4, 4, 4]})
    d = _data(ggplot(df, aes(y="v")).add(geom_bar()))
    assert d["y"].tolist() == [1, 2, 4] and d["x"].tolist() == [2, 1, 3]
    np.testing.assert_allclose(d["ymin"], [0.55, 1.55, 3.55])
    np.testing.assert_allclose(d["ymax"], [1.45, 2.45, 4.45])


def test_layer_and_plot_mapping_and_x_title():
    a = ggplot(CUT).add(geom_bar(aes(y="cut"))).build()
    b = ggplot(CUT, aes(y="cut")).add(geom_bar()).build()
    c = ggplot(CUT, aes(y="cut")).add(stat_count()).build()
    for m in (a, b, c):
        assert m.layer_data[0]["x"].tolist() == [2, 3, 1]
        assert m.layer_data[0]["flipped_aes"].all()
    assert b.panel.x_title.text == "count"


def test_stacked_horizontal_bars_stack_along_x():
    # ggplot(d, aes(y = cut, fill = g)) + geom_bar(): first group ends furthest out
    d = _data(ggplot(CUT, aes(y="cut", fill="g")).add(geom_bar()))
    d = d.sort_values(["y", "group"]).reset_index(drop=True)
    assert d["y"].tolist() == [1, 1, 2, 2, 3]
    np.testing.assert_allclose(d["xmin"], [1, 0, 1, 0, 0])
    np.testing.assert_allclose(d["xmax"], [2, 1, 3, 1, 1])
    np.testing.assert_allclose(d["ymax"] - d["ymin"], 0.9)


def test_explicit_orientation_and_after_stat_prop():
    d = _data(
        ggplot(CUT).add(geom_bar(aes(y="cut", x=after_stat("prop"), group=1), orientation="y"))
    )
    np.testing.assert_allclose(d["x"], [2 / 6, 3 / 6, 1 / 6])
    assert d["flipped_aes"].all()
    d = _data(ggplot(CUT, aes("cut")).add(geom_bar(orientation="x")))
    assert d["y"].tolist() == [2, 3, 1] and not d["flipped_aes"].any()


def test_stat_count_flipped_compute():
    out = StatCount().compute(pd.DataFrame({"y": [1.0, 1.0, 2.0]}))
    assert out["y"].tolist() == [1, 2] and out["count"].tolist() == [2, 1]
    assert out["flipped_aes"].all() and "x" not in out.columns
    out = StatCount(orientation="y").compute(pd.DataFrame({"y": [3.0, 3.0]}))
    assert out["count"].tolist() == [2]


def test_x_and_y_together_is_an_error():
    with pytest.raises(BuildError, match=r"stat_count\(\) must only have an x or y"):
        ggplot(CUT, aes("cut", "w")).add(geom_bar()).build()
    with pytest.raises(BuildError, match=r"stat_bin\(\) must only have an x or y"):
        ggplot(CUT, aes("w", "w")).add(geom_histogram(bins=3)).build()
    with pytest.raises(ValueError, match="requires an x or y"):
        StatCount().compute(pd.DataFrame({"weight": [1.0]}))


# -- stat_bin ----------------------------------------------------------------------


def test_histogram_on_y_matches_ggplot2():
    # ggplot(data.frame(v = V), aes(y = v)) + geom_histogram(bins = 5)
    d = _data(ggplot(pd.DataFrame({"v": V}), aes(y="v")).add(geom_histogram(bins=5)))
    np.testing.assert_allclose(d["y"], [0.5, 2.775, 5.05, 7.325, 9.6])
    np.testing.assert_allclose(d["ymin"], [-0.6375, 1.6375, 3.9125, 6.1875, 8.4625])
    np.testing.assert_allclose(d["ymax"], [1.6375, 3.9125, 6.1875, 8.4625, 10.7375])
    assert d["count"].tolist() == [2, 3, 3, 1, 1] and d["x"].tolist() == [2, 3, 3, 1, 1]
    np.testing.assert_allclose(
        d["density"], [0.08791209, 0.13186813, 0.13186813, 0.04395604, 0.04395604]
    )
    np.testing.assert_allclose(d["xmin"], 0)
    assert d["flipped_aes"].all()


def test_freqpoly_and_stat_bin_on_y():
    # ggplot(data.frame(v = V), aes(y = v)) + geom_freqpoly(binwidth = 2)
    d = _data(ggplot(pd.DataFrame({"v": V}), aes(y="v")).add(geom_freqpoly(binwidth=2)))
    np.testing.assert_allclose(d["y"], [-2, 0, 2, 4, 6, 8, 10, 12])
    assert d["x"].tolist() == [0, 1, 3, 3, 1, 1, 1, 0]
    d = _data(ggplot(pd.DataFrame({"v": V}), aes(y="v")).add(stat_bin(binwidth=2)))
    assert d["x"].tolist() == [1, 3, 3, 1, 1, 1]


def test_stat_bin_flipped_uses_y_scale_limits():
    out = StatBin(binwidth=1).compute(pd.DataFrame({"y": [0.0, 1.0, 1.0]}))
    assert out["count"].tolist() == [1, 2]
    np.testing.assert_allclose(out["ymin"], [-0.5, 0.5])
    assert out["flipped_aes"].all() and "xmin" not in out.columns


# -- after_stat defaults reach make_layer ------------------------------------------


def test_stat_defaults_added_by_make_layer():
    layer = geom_point(aes("cut"), stat="count")
    assert layer.mapping["y"] == after_stat("count")
    assert isinstance(layer.mapping["y"], StatDefault) and layer.mapping["x"] == "cut"
    d = _data(ggplot(CUT).add(geom_point(aes("cut"), stat="count")))
    assert d["y"].tolist() == [2, 3, 1]
    # a constant or explicit None is not overridden
    assert "y" not in geom_bar(aes("cut"), y=1).mapping
    assert geom_bar(aes("cut", y=None)).mapping["y"] is None


def test_user_after_stat_beats_plot_mapping():
    # an explicit after_stat in the layer still overrides the inherited aesthetic
    df = CUT.assign(n=1.0)
    d = _data(ggplot(df, aes("cut", y="n")).add(geom_bar(aes(y=after_stat("count")))))
    assert d["y"].tolist() == [2, 3, 1] and not d["flipped_aes"].any()


# -- bins warning --------------------------------------------------------------------


def test_bins_warning_once_per_build_under_facets():
    df = pd.DataFrame({"v": np.arange(12.0), "f": list("abc") * 4})
    p = ggplot(df, aes("v")).add(geom_histogram()).add(facet_wrap("f"))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        p.build()
    assert sum("bins = 30" in str(w.message) for w in rec) == 1
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ggplot(df, aes(y="v")).add(geom_histogram()).build()
    assert sum("bins = 30" in str(w.message) for w in rec) == 1
