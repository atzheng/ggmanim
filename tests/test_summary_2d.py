"""stat_summary_2d / stat_summary_hex, density_2d weights, and titles clearing
a tall legend (ggm-7ww)."""

import numpy as np
import pandas as pd
import pytest

from ggmanim import (
    aes,
    geom_density_2d_filled,
    geom_point,
    ggplot,
    labs,
    stat_summary2d,
    stat_summary_2d,
    stat_summary_hex,
    theme,
)
from ggmanim.geoms import GeomHex, GeomTile
from ggmanim.registry import COMPONENTS
from ggmanim.stats import (
    STATS,
    StatBin2d,
    StatBinHex,
    StatDensity2d,
    StatSummary2d,
    StatSummaryHex,
    kde2d,
)


def cloud(n=600, seed=4):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})
    df["z"] = df["x"] * 2 + df["y"]
    return df


# -- registry ----------------------------------------------------------------
def test_registered_and_paired_geoms():
    for n in ("stat_summary_2d", "stat_summary2d", "stat_summary_hex"):
        assert n in COMPONENTS
    assert STATS["summary_2d"] is StatSummary2d
    assert STATS["summary_hex"] is StatSummaryHex
    assert isinstance(stat_summary_2d().geom, GeomTile)
    assert isinstance(stat_summary2d().geom, GeomTile)
    assert isinstance(stat_summary_hex().geom, GeomHex)


# -- stat_summary_2d ---------------------------------------------------------
def test_summary_2d_mean_matches_manual():
    df = cloud()
    out = StatSummary2d(bins=6).compute(df)
    assert {"xmin", "xmax", "ymin", "ymax", "x", "y", "width", "height", "value"} <= set(
        out.columns
    )
    for row in out.sample(5, random_state=0).itertuples():
        sel = (df.x > row.xmin) & (df.x <= row.xmax) & (df.y > row.ymin) & (df.y <= row.ymax)
        assert row.value == pytest.approx(df.z[sel].mean())


def test_summary_2d_length_equals_bin2d_count():
    df = cloud()
    a = StatSummary2d(bins=8, fun="length").compute(df)
    b = StatBin2d(bins=8).compute(df)
    m = a.merge(b, on=["xmin", "ymin"])
    assert len(m) == len(a) == len(b)
    assert np.allclose(m["value"], m["count"])


def test_summary_2d_fun_callable_and_args_and_drop():
    df = cloud()
    q = StatSummary2d(bins=5, fun=np.quantile, fun_args={"q": 0.9}).compute(df)
    assert np.isfinite(q["value"]).all()
    full = StatSummary2d(bins=5, drop=False).compute(df)
    kept = StatSummary2d(bins=5).compute(df)
    assert len(full) > len(kept)
    assert full["value"].notna().sum() == len(kept)


def test_summary_2d_builds_tiles_with_value_fill():
    g = ggplot(cloud(), aes("x", "y", z="z")).add(stat_summary_2d(bins=6)).build()
    marks = g.layers[0].marks
    assert len(marks) > 10
    assert len({m.get_fill_color().to_hex() for m in marks}) > 3


# -- stat_summary_hex --------------------------------------------------------
def test_summary_hex_length_equals_bin_hex_count():
    df = cloud()
    a = StatSummaryHex(bins=10, fun="length").compute(df)
    b = StatBinHex(bins=10).compute(df)
    assert len(a) == len(b)
    assert np.allclose(a[["x", "y"]].to_numpy(), b[["x", "y"]].to_numpy())
    assert np.allclose(a["value"], b["count"])
    assert np.allclose(a["width"], b["width"]) and np.allclose(a["height"], b["height"])


def test_summary_hex_mean_and_median():
    df = cloud()
    n = StatBinHex(bins=10).compute(df)["count"].to_numpy()
    s = StatSummaryHex(bins=10, fun="sum").compute(df)["value"].to_numpy()
    m = StatSummaryHex(bins=10).compute(df)["value"].to_numpy()
    assert np.allclose(m, s / n)
    med = StatSummaryHex(bins=10, fun="median").compute(df)
    assert np.isfinite(med["value"]).all()


def test_summary_hex_builds_hexagons():
    g = ggplot(cloud(), aes("x", "y", z="z")).add(stat_summary_hex(bins=8)).build()
    assert len(g.layers[0].marks) > 10


# -- density_2d weights ------------------------------------------------------
def test_kde2d_integer_weights_equal_repeated_rows():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=40), rng.normal(size=40)
    w = rng.integers(1, 4, size=40)
    h = (1.0, 1.2)
    _, _, zw = kde2d(x, y, h=h, n=15, lims=(-3, 3, -3, 3), weights=w)
    _, _, zr = kde2d(np.repeat(x, w), np.repeat(y, w), h=h, n=15, lims=(-3, 3, -3, 3))
    assert np.allclose(zw, zr)


def test_kde2d_rejects_bad_weights():
    with pytest.raises(ValueError):
        kde2d([0, 1, 2], [0, 1, 2], h=1, weights=[1, -1, 1])
    with pytest.raises(ValueError):
        kde2d([0, 1, 2], [0, 1, 2], h=1, weights=[0, 0, 0])


def test_stat_density_2d_weight_shifts_mass():
    df = pd.DataFrame({"x": [0.0, 0.0, 4.0, 4.0], "y": [0.0, 0.1, 4.0, 4.1]})
    plain = StatDensity2d(contour=False, n=21, h=2).compute(df)
    heavy = StatDensity2d(contour=False, n=21, h=2).compute(df.assign(weight=[9, 9, 1, 1]))
    near = (plain.x < 1) & (plain.y < 1)
    assert heavy.density[near].sum() > plain.density[near].sum()
    # unit weights change nothing
    ones = StatDensity2d(contour=False, n=21, h=2).compute(df.assign(weight=1.0))
    assert np.allclose(ones.density, plain.density)


def test_geom_density_2d_filled_accepts_weight():
    df = cloud(200).assign(w=np.linspace(0.1, 2, 200))
    g = ggplot(df, aes("x", "y", weight="w")).add(geom_density_2d_filled(bins=4)).build()
    assert len(g.layers[0].marks) >= 1


# -- titles clear a legend taller than the panel ------------------------------
@pytest.mark.parametrize("subtitle", [None, "a subtitle"])
def test_title_above_tall_legend(subtitle):
    df = pd.DataFrame({"x": np.arange(12.0), "y": np.arange(12.0), "g": list("abcdefghijkl")})
    p = (
        ggplot(df, aes("x", "y", colour="g"))
        .add(geom_point())
        .add(labs(title="A long plot title spanning the figure", subtitle=subtitle, caption="c"))
        .add(theme(panel_width=3.0, panel_height=1.5))
    )
    g = p.build()
    assert g.legend.get_top()[1] > g.panel.get_top()[1]  # the legend is taller
    lowest = g.subtitle if subtitle is not None else g.title
    assert lowest.get_bottom()[1] > g.legend.get_top()[1]
    assert g.title.get_bottom()[1] > lowest.get_bottom()[1] - 1e-9
    assert g.caption.get_top()[1] < g.legend.get_bottom()[1]
