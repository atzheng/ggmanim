"""stat_boxplot / geom_boxplot (fixtures from tests/fixtures/distributions_r.R)."""

from pathlib import Path

import numpy as np
import pandas as pd
from manim import VGroup

import ggmanim
from ggmanim import aes, coord_flip, geom_boxplot, geom_jitter, ggplot, stat_boxplot
from ggmanim.geoms import GeomBoxplot
from ggmanim.registry import COMPONENTS
from ggmanim.stats import StatBoxplot
from ggmanim.stats.boxplot import boxplot_stats, weighted_quantile

FIX = Path(__file__).parent / "fixtures"
DATA = pd.read_csv(FIX / "dist_data.csv")


def _ld(plot, i=0):
    return plot.build().layer_data[i]


def test_registered_and_exported():
    for name in ("geom_boxplot", "stat_boxplot"):
        assert name in COMPONENTS and name in ggmanim.__all__
    assert isinstance(geom_boxplot().geom, GeomBoxplot)
    assert isinstance(geom_boxplot().stat, StatBoxplot)
    assert isinstance(stat_boxplot().geom, GeomBoxplot)


def test_stats_match_ggplot2():
    ref = pd.read_csv(FIX / "dist_boxplot_r.csv", keep_default_na=False)
    ld = _ld(ggplot(DATA, aes("g", "v")).add(geom_boxplot()))
    cols = ["x", "ymin", "lower", "middle", "upper", "ymax", "notchlower", "notchupper"]
    np.testing.assert_allclose(ld[cols].to_numpy(float), ref[cols].to_numpy(float), rtol=1e-9)
    # ggplot2 4.0: boxes are 0.75 wide on a discrete scale
    np.testing.assert_allclose(ld[["xmin", "xmax"]].to_numpy(float), ref[["xmin", "xmax"]])
    for mine, theirs in zip(ld["outliers"], ref["outliers"]):
        want = [float(s) for s in theirs.split(";")] if theirs else []
        np.testing.assert_allclose(sorted(mine), sorted(want), rtol=1e-9)


def test_quantiles_and_whisker_rule():
    rng = np.random.default_rng(0)
    y = np.append(rng.normal(size=50), [8.0, -9.0])
    stats, out, n = boxplot_stats(y)
    q = np.quantile(y, [0.25, 0.5, 0.75])
    assert (stats["lower"], stats["middle"], stats["upper"]) == tuple(q)
    iqr = q[2] - q[0]
    inside = y[(y >= q[0] - 1.5 * iqr) & (y <= q[2] + 1.5 * iqr)]
    assert stats["ymin"] == inside.min() and stats["ymax"] == inside.max()
    assert sorted(out) == [-9.0, 8.0] and n == 52
    # no outliers: whiskers are the extremes
    stats, out, _ = boxplot_stats(np.arange(10.0))
    assert stats["ymin"] == 0 and stats["ymax"] == 9 and len(out) == 0
    # coef widens the whiskers
    stats, out, _ = boxplot_stats(y, coef=100)
    assert stats["ymax"] == 8.0 and len(out) == 0


def test_weighted_quantiles():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert list(weighted_quantile(y, np.array([1, 1, 1, 5.0]), [0, 0.5, 1])) == [1, 4, 4]
    assert list(weighted_quantile(y, np.ones(4), [0.25, 0.75])) == [1, 3]
    df = pd.DataFrame({"x": ["a"] * 4, "y": y, "w": [1, 1, 1, 5]})
    ld = _ld(ggplot(df, aes("x", "y", weight="w")).add(geom_boxplot()))
    assert ld["middle"][0] == 4 and ld["n"][0] == 8


def test_dodged_by_fill_and_outliers_train_y():
    df = DATA.assign(s=np.where(np.arange(len(DATA)) % 2, "m", "f"))
    p = ggplot(df, aes("g", "v", fill="s")).add(geom_boxplot())
    ld = _ld(p)
    assert len(ld) == 6
    left, right = ld.iloc[0::2], ld.iloc[1::2]
    assert (left["xmax"].to_numpy() < right["xmin"].to_numpy()).all()
    np.testing.assert_allclose((left["x"].to_numpy() + right["x"].to_numpy()) / 2, [1, 2, 3])
    built = p.build()
    lo, hi = built.scales["y"].get_limits()
    assert lo <= DATA["v"].min() and hi >= DATA["v"].max()
    assert "ymax_final" in ld.columns
    mob = built.layers[0]
    assert len(mob.marks) == 6 and all(isinstance(m, VGroup) for m in mob.marks)


def test_outliers_false_and_outlier_params():
    p = ggplot(DATA, aes("g", "v")).add(geom_boxplot(outliers=False))
    ld = _ld(p)
    assert "outliers" not in ld.columns
    _, hi = p.build().scales["y"].get_limits()
    assert hi < 12
    m1 = ggplot(DATA, aes("g", "v")).add(geom_boxplot()).build().layers[0].marks[1]
    m2 = (
        ggplot(DATA, aes("g", "v"))
        .add(geom_boxplot(outlier_colour="red", outlier_size=0.1))
        .build()
        .layers[0]
        .marks[1]
    )
    assert len(m1) == 3 and len(m2) == 3  # outliers, whiskers, box
    assert len(m1[0]) == 2  # two outlier points in group b
    assert m2[0][0].width > m1[0][0].width


def test_continuous_x_one_box_per_group_and_width():
    df = pd.DataFrame({"x": np.repeat([1.0, 2.0, 3.0], 5), "y": np.arange(15.0)})
    ld = _ld(ggplot(df, aes("x", "y")).add(geom_boxplot()))
    assert len(ld) == 1 and ld["x"][0] == 2.0
    np.testing.assert_allclose(ld["xmax"] - ld["xmin"], 0.9 * 2.0)
    ld = _ld(ggplot(df, aes("x", "y", group="x")).add(geom_boxplot()))
    assert len(ld) == 3
    np.testing.assert_allclose(ld["xmax"] - ld["xmin"], 0.75)


def test_varwidth_notch_staples():
    p = ggplot(DATA, aes("g", "v")).add(geom_boxplot(varwidth=True))
    ld = _ld(p)
    widths = (ld["xmax"] - ld["xmin"]).to_numpy()
    n = DATA.groupby("g").size().to_numpy()
    np.testing.assert_allclose(widths, 0.75 * np.sqrt(n) / np.sqrt(n).max())
    m = ggplot(DATA, aes("g", "v")).add(geom_boxplot(notch=True, staplewidth=0.5)).build()
    mark = m.layers[0].marks[2]  # outliers, staples, whiskers, box
    assert len(mark) == 4
    assert len(mark[3][0].get_vertices()) >= 10  # notched polygon


def test_horizontal_boxplot():
    ld = _ld(ggplot(DATA, aes("v", "g")).add(geom_boxplot()))
    ref = pd.read_csv(FIX / "dist_boxplot_r.csv", keep_default_na=False)
    np.testing.assert_allclose(ld["xmiddle"], ref["middle"], rtol=1e-9)
    np.testing.assert_allclose(ld[["ymin", "ymax"]].to_numpy(float), ref[["xmin", "xmax"]])
    assert ld["flipped_aes"].all()
    mob = ggplot(DATA, aes("v", "g")).add(geom_boxplot()).build()
    assert len(mob.layers[0].marks) == 3
    # orientation can also be forced, and coord_flip draws vertical data sideways
    ggplot(DATA, aes("g", "v")).add(geom_boxplot(), coord_flip()).build()


def test_with_jitter_and_key():
    p = (
        ggplot(DATA, aes("g", "v", fill="g"))
        .add(geom_boxplot(alpha=0.5))
        .add(geom_jitter(width=0.1))
        .build()
    )
    assert len(p.layers[0].marks) == 3
    key = GeomBoxplot().draw_key({"fill": "#ff0000"})
    assert key.width > 0


def test_empty_panel_and_single_value():
    df = pd.DataFrame({"x": ["a"], "y": [1.0]})
    ld = _ld(ggplot(df, aes("x", "y")).add(geom_boxplot()))
    assert ld["ymin"][0] == ld["ymax"][0] == 1.0
    assert GeomBoxplot().draw_panel(ld.iloc[0:0], None, _Ident()).marks == []


class _Ident:
    clip = "off"

    def transform(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        return np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])


def test_horizontal_boxplots_dodge_along_y():
    df = DATA.assign(s=np.where(np.arange(len(DATA)) % 2, "m", "f"))
    ld = _ld(ggplot(df, aes("v", "g", fill="s")).add(geom_boxplot()))
    vert = _ld(ggplot(df, aes("g", "v", fill="s")).add(geom_boxplot()))
    np.testing.assert_allclose(ld["y"], vert["x"])
    np.testing.assert_allclose(ld[["ymin", "ymax"]].to_numpy(), vert[["xmin", "xmax"]].to_numpy())
    np.testing.assert_allclose(ld[["xmin", "xmax"]].to_numpy(), vert[["ymin", "ymax"]].to_numpy())
