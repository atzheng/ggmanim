"""stat_bin / geom_histogram / geom_freqpoly; bin edges as ggplot2's."""

import warnings

import numpy as np
import pandas as pd
import pytest

from ggmanim import (
    aes,
    after_stat,
    geom_freqpoly,
    geom_histogram,
    ggplot,
    scale_x_log10,
    stat_bin,
)
from ggmanim.geoms import GeomBar, GeomPath
from ggmanim.stats import STATS, StatBin
from ggmanim.stats.bin import BINS_WARNING, bin_breaks_bins, bin_breaks_width

# (x_range, kwargs, expected edges): ggplot2 (main, R/bin.R) bin_breaks_width / bin_breaks_bins
WIDTH_CASES = [
    ((0, 10), {"width": 1}, np.arange(-0.5, 10.6, 1)),  # boundary = width / 2
    ((0, 10), {"width": 1, "boundary": 0}, np.arange(0, 10.1, 1)),
    ((1, 9), {"width": 2, "center": 0}, [1, 3, 5, 7, 9]),
    ((0, 20), {"width": 10, "boundary": 0}, [0, 10, 20]),  # no extra bin at max
    ((0.3, 2.9), {"width": 0.5}, [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25]),
    ((-3, 3), {"width": 2, "boundary": 1}, [-3, -1, 1, 3]),
    ((5, 5), {"width": 1}, [4.5, 5.5]),
]
BINS_CASES = [
    ((0, 29), {"bins": 30}, np.arange(-0.5, 29.6, 1)),
    ((0, 9), {"bins": 10}, np.arange(-0.5, 9.6, 1)),
    ((0, 10), {"bins": 1}, [0, 10]),
    # range coincides with the boundary: exactly `bins` bins (current ggplot2)
    ((0, 10), {"bins": 5, "boundary": 0}, [0, 2, 4, 6, 8, 10]),
    ((0, 10), {"bins": 5, "boundary": 1}, [-1.5, 1, 3.5, 6, 8.5, 11]),
    ((0, 10), {"bins": 5, "center": 0}, [-1.25, 1.25, 3.75, 6.25, 8.75, 11.25]),
    ((5, 5), {"bins": 30}, [4.95, 5.05]),  # zero range: width 0.1
]


@pytest.mark.parametrize("rng, kw, expected", WIDTH_CASES)
def test_bin_breaks_width_matches_ggplot2(rng, kw, expected):
    np.testing.assert_allclose(bin_breaks_width(rng, **kw), expected, atol=1e-9)


@pytest.mark.parametrize("rng, kw, expected", BINS_CASES)
def test_bin_breaks_bins_matches_ggplot2(rng, kw, expected):
    np.testing.assert_allclose(bin_breaks_bins(rng, **kw), expected, atol=1e-9)


def test_center_and_boundary_exclusive():
    with pytest.raises(ValueError, match="only one"):
        bin_breaks_width((0, 1), 0.5, center=0, boundary=0)


def test_closed_right_and_left():
    df = pd.DataFrame({"x": [0.0, 1.0, 1.0, 2.0]})
    right = StatBin(breaks=[0, 1, 2]).compute(df)
    assert right["count"].tolist() == [3, 1]  # [0,1], (1,2]
    left = StatBin(breaks=[0, 1, 2], closed="left").compute(df)
    assert left["count"].tolist() == [1, 3]  # [0,1), [1,2]


def test_output_columns_density_and_weight():
    df = pd.DataFrame({"x": [0.0, 1.0, 1.0, 2.0, 3.0]})
    out = StatBin(binwidth=1).compute(df)
    assert out["count"].tolist() == [1, 2, 1, 1]
    np.testing.assert_allclose(out["x"], [0, 1, 2, 3])
    np.testing.assert_allclose(out["xmin"], [-0.5, 0.5, 1.5, 2.5])
    np.testing.assert_allclose(out["width"], 1)
    np.testing.assert_allclose((out["density"] * out["width"]).sum(), 1)
    np.testing.assert_allclose(out["ncount"], [0.5, 1, 0.5, 0.5])
    np.testing.assert_allclose(out["ndensity"], [0.5, 1, 0.5, 0.5])
    assert {"flipped_aes", "y"} <= set(out.columns)
    out = StatBin(binwidth=1).compute(df.assign(weight=[2, 1, 1, 0, np.nan]))
    assert out["count"].tolist() == [2, 2, 0, 0]
    assert "weight" not in out.columns


def test_pad_adds_empty_bins():
    out = StatBin(binwidth=1, pad=True).compute(pd.DataFrame({"x": [0.0, 1.0]}))
    assert out["count"].tolist() == [0, 1, 1, 0]
    np.testing.assert_allclose(out["x"], [-1, 0, 1, 2])


def test_default_bins_warns():
    df = pd.DataFrame({"x": np.arange(10.0)})
    with pytest.warns(UserWarning, match="bins = 30"):
        d = ggplot(df, aes("x")).add(geom_histogram()).build().layer_data[0]
    assert len(d) == 30 and BINS_WARNING.startswith("stat_bin()")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ggplot(df, aes("x")).add(geom_histogram(bins=5)).build()


def test_histogram_layers():
    assert STATS["bin"] is StatBin
    h = geom_histogram(binwidth=1)
    assert isinstance(h.geom, GeomBar) and type(h.position).__name__ == "PositionStack"
    assert h.stat.params["binwidth"] == 1 and h.mapping["y"] == after_stat("count")
    f = geom_freqpoly(bins=10)
    assert isinstance(f.geom, GeomPath) and f.stat.params["pad"] is True
    assert type(f.position).__name__ == "PositionIdentity"


def test_stacked_histogram_by_fill():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=200), "g": np.repeat(["a", "b"], 100)})
    m = ggplot(df, aes("x", fill="g")).add(geom_histogram(binwidth=0.5)).build()
    d = m.layer_data[0]
    assert set(d["group"]) == {1, 2}
    # both groups share the edges; stacked tops are the total counts
    tops = d.groupby("x")["ymax"].max()
    totals = d.groupby("x")["count"].sum()
    np.testing.assert_allclose(tops.to_numpy(), totals.to_numpy())
    assert totals.sum() == 200
    assert m.panel.y_title.text == "count"


def test_after_stat_density():
    df = pd.DataFrame({"x": [0.0, 1.0, 1.0, 2.0, 3.0]})
    m = ggplot(df, aes("x")).add(geom_histogram(aes(y=after_stat("density")), binwidth=1)).build()
    d = m.layer_data[0]
    np.testing.assert_allclose(d["y"], [0.2, 0.4, 0.2, 0.2])
    assert m.panel.y_title.text == "density"


def test_breaks_param_in_data_units_on_log_x():
    df = pd.DataFrame({"x": [1.0, 5.0, 20.0, 50.0, 500.0]})
    p = ggplot(df, aes("x")).add(scale_x_log10())
    d = p.add(geom_histogram(breaks=[1, 10, 100, 1000])).build().layer_data[0]
    np.testing.assert_allclose(d["xmin"], [0, 1, 2])
    assert d["count"].tolist() == [2, 2, 1]
    d = p.add(geom_histogram(binwidth=1, boundary=0)).build().layer_data[0]
    np.testing.assert_allclose(d["width"], 1)  # equal bins in log units


def test_freqpoly_lines_per_group():
    df = pd.DataFrame({"x": [0.0, 1.0, 1.0, 2.0, 0.0, 2.0], "g": list("aaaabb")})
    m = ggplot(df, aes("x", color="g")).add(geom_freqpoly(binwidth=1)).build()
    d = m.layer_data[0]
    a = d[d["group"] == 1]
    assert a["y"].tolist() == [0, 1, 2, 1, 0]
    assert len(m.layers[0].marks) == 2


def test_discrete_x_rejected_and_stat_bin_layer():
    with pytest.raises(Exception, match="continuous x"):
        ggplot(pd.DataFrame({"x": list("abc")}), aes("x")).add(geom_histogram(bins=3)).build()
    df = pd.DataFrame({"x": np.arange(10.0)})
    m = ggplot(df, aes("x")).add(stat_bin(bins=5)).build()
    assert len(m.layers[0].marks) == 5
