"""stat_summary / stat_summary_bin and the fun_data helpers."""

import numpy as np
import pandas as pd
import pytest

import ggmanim
from ggmanim import aes, ggplot, mean_cl_normal, mean_sdl, mean_se, median_hilow, stat_summary
from ggmanim.geoms import GeomCrossbar, GeomErrorbar, GeomPointrange
from ggmanim.registry import COMPONENTS
from ggmanim.stats import STATS, StatSummary, StatSummaryBin, stat_summary_bin
from ggmanim.stats.summary import bin_breaks, summary_function

RNG = np.random.default_rng(0)
DF = pd.DataFrame({"x": np.repeat([1.0, 2.0, 3.0], 10), "y": RNG.normal(5, 2, 30)})


def test_registered():
    for n in ("stat_summary", "stat_summary_bin", "mean_se", "mean_cl_normal"):
        assert n in ggmanim.__all__
    assert "stat_summary" in COMPONENTS
    assert STATS["summary"] is StatSummary and STATS["summary_bin"] is StatSummaryBin


def test_mean_se_matches_numpy():
    out = StatSummary().compute(DF)
    for x, g in DF.groupby("x"):
        row = out[out.x == x].iloc[0]
        m, se = g.y.mean(), g.y.std(ddof=1) / np.sqrt(len(g))
        assert row.y == pytest.approx(m)
        assert row.ymin == pytest.approx(m - se)
        assert row.ymax == pytest.approx(m + se)


def test_fun_data_helpers():
    x = np.array([1.0, 2.0, 4.0, 7.0, np.nan])
    v = x[:4]
    sd = np.std(v, ddof=1)
    assert mean_se(x, mult=2)["ymax"] == pytest.approx(v.mean() + 2 * sd / 2)
    assert mean_sdl(x)["ymin"] == pytest.approx(v.mean() - 2 * sd)
    ci = mean_cl_normal(x)
    try:
        from scipy import stats

        q = stats.t.ppf(0.975, 3)
        assert ci["ymax"] == pytest.approx(v.mean() + q * sd / 2)
    except ImportError:  # pragma: no cover
        assert ci["ymax"] > v.mean()
    hl = median_hilow(x, conf_int=0.5)
    assert (hl["ymin"], hl["y"], hl["ymax"]) == pytest.approx(
        tuple(np.quantile(v, [0.25, 0.5, 0.75]))
    )
    assert np.isnan(mean_se([3.0])["ymin"])  # se of one value is undefined


def test_fun_fun_min_fun_max_and_names():
    s = summary_function(fun="median", fun_min=np.min, fun_max=lambda v: v.max())
    assert s(np.array([1.0, 5.0, 2.0])) == {"y": 2.0, "ymin": 1.0, "ymax": 5.0}
    assert summary_function(fun="mean")(np.array([1.0, 3.0])) == {"y": 2.0}
    with pytest.raises(ValueError, match="unknown"):
        summary_function(fun_data="mean_magic")
    # fun_args reach the function; a callable fun_data may return a Series
    s = summary_function(fun_data=lambda v, k: pd.Series({"y": v.mean() * k}), fun_args={"k": 2})
    assert s(np.array([1.0, 3.0])) == {"y": 4.0}


def test_layer_default_geom_pointrange_and_build():
    layer = stat_summary(fun_data="mean_sdl")
    assert isinstance(layer.geom, GeomPointrange) and isinstance(layer.stat, StatSummary)
    assert layer.stat.params["fun_data"] == "mean_sdl"
    m = ggplot(DF, aes("x", "y")).stat_summary().build()
    marks = m.layers[0].marks
    assert len(marks) == 3
    for (x, g), (_line, dot) in zip(DF.groupby("x"), marks):
        np.testing.assert_allclose(dot.get_center(), m.panel.transform(x, g.y.mean())[0], atol=1e-9)


def test_other_geoms_and_stat_string():
    m = ggplot(DF, aes("x", "y")).stat_summary(geom="errorbar", width=0.3).build()
    assert len(m.layers[0].marks) == 3
    assert isinstance(stat_summary(geom="crossbar").geom, GeomCrossbar)
    layer = ggmanim.geom_errorbar(stat="summary", fun_data="median_hilow")
    assert isinstance(layer.geom, GeomErrorbar) and layer.stat.params["fun_data"] == "median_hilow"
    with pytest.raises(ValueError, match="unknown geom"):
        stat_summary(geom="nonsense")


def test_discrete_x_and_constant_columns_carried():
    df = pd.DataFrame({"g": list("abab"), "v": [1.0, 2.0, 3.0, 4.0], "k": ["p", "q", "p", "q"]})
    out = StatSummary(fun="mean").compute(df.rename(columns={"g": "x", "v": "y"}).assign(
        x=lambda d: d.x.map({"a": 1.0, "b": 2.0})))  # fmt: skip
    assert list(out.x) == [1.0, 2.0] and list(out.y) == [2.0, 3.0]
    assert list(out.k) == ["p", "q"]  # constant within each x bucket
    m = ggplot(df, aes("g", "v", key="k")).stat_summary().build()
    assert m.layers[0].keys == ["p", "q"]


def test_groups_summarised_separately():
    df = pd.concat([DF.assign(c="u"), DF.assign(y=DF.y + 10, c="v")])
    out = ggplot(df, aes("x", "y", color="c")).stat_summary().build()
    assert len(out.layers[0].marks) == 6


def test_summary_bin():
    x = np.linspace(0, 10, 101)
    df = pd.DataFrame({"x": x, "y": 2 * x})
    out = StatSummaryBin(bins=5, fun="mean").compute(df)
    assert len(out) == 5
    np.testing.assert_allclose(out.x, [1, 3, 5, 7, 9])
    np.testing.assert_allclose(out.width, 2.0)
    first = df[(df.x >= 0) & (df.x < 2)]
    assert out.y.iloc[0] == pytest.approx(first.y.mean())
    assert out.y.iloc[-1] == pytest.approx(df[df.x >= 8].y.mean())  # right edge included
    np.testing.assert_allclose(bin_breaks(0, 10, binwidth=3), [0, 3, 6, 9, 12])
    np.testing.assert_allclose(bin_breaks(0, 1, breaks=[1, 0, 0.5]), [0, 0.5, 1])
    m = ggplot(df, aes("x", "y")).add(stat_summary_bin(binwidth=2.5, geom="crossbar")).build()
    assert len(m.layers[0].marks) == 4
    box, _ = m.layers[0].marks[0]
    w = box.get_vertices()[:, 0]
    assert w.max() - w.min() == pytest.approx(
        m.panel.transform(2.5, 0)[0][0] - m.panel.transform(0, 0)[0][0]
    )
