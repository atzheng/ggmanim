"""stat_bin_2d / geom_bin_2d: breaks like ggplot2 4.0, counts, weights, drawing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ggmanim import aes, geom_bin2d, geom_bin_2d, ggplot, stat_bin_2d
from ggmanim.geoms import GeomBin2d
from ggmanim.registry import COMPONENTS
from ggmanim.stats import StatBin2d, bin2d_breaks

FIX = Path(__file__).parent / "fixtures"
DATA = pd.read_csv(FIX / "density2d_data.csv")
GG_R = pd.read_csv(FIX / "ggplot2d_r.csv")


def cloud(n=500, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})


def test_registered():
    for n in ("geom_bin_2d", "geom_bin2d", "stat_bin_2d", "stat_bin2d"):
        assert n in COMPONENTS
    assert geom_bin2d is geom_bin_2d


def test_counts_sum_to_n():
    out = StatBin2d().compute(cloud())
    assert out["count"].sum() == 500
    assert (out["count"] > 0).all()
    assert out["density"].sum() == pytest.approx(1.0)
    assert out["ncount"].max() == pytest.approx(1.0)
    assert out["ndensity"].max() == pytest.approx(1.0)
    assert np.allclose(out["x"], (out["xmin"] + out["xmax"]) / 2)
    assert np.allclose(out["width"], out["xmax"] - out["xmin"])


def test_matches_ggplot2():
    out = StatBin2d(bins=8).compute(DATA[["x", "y"]])
    ref = GG_R[GG_R["kind"] == "bin2d"]
    ours = {(round(a, 6), round(b, 6)): c for a, b, c in zip(out["x"], out["y"], out["count"])}
    theirs = {(round(a, 6), round(b, 6)): c for a, b, c in zip(ref["a"], ref["b"], ref["c"])}
    assert ours == theirs


def test_breaks_rules():
    # bins: width range / (bins - 1) aligned on 0
    b = bin2d_breaks(0.3, 9.7, bins=5)
    w = (9.7 - 0.3) / 4
    assert np.allclose(np.diff(b), w)
    assert b[0] <= 0.3 and b[-1] >= 9.7
    assert np.isclose((b[0] / w) % 1, 0) or np.isclose((b[0] / w) % 1, 1)
    # a range end on a bin edge: width range / bins
    assert np.allclose(np.diff(bin2d_breaks(0.0, 10.0, bins=5)), 2.0)
    # binwidth with boundary / center
    assert bin2d_breaks(0.2, 2.9, binwidth=1, boundary=0).tolist() == [0, 1, 2, 3]
    assert bin2d_breaks(0.2, 2.9, binwidth=1, boundary=None, center=0).tolist() == [
        -0.5,
        0.5,
        1.5,
        2.5,
        3.5,
    ]
    assert bin2d_breaks(0, 1, breaks=[1, 0, 0.5]).tolist() == [0, 0.5, 1]
    with pytest.raises(ValueError):
        bin2d_breaks(0, 1, binwidth=1, boundary=0, center=0)


def test_closed_and_edges():
    df = pd.DataFrame({"x": [0.0, 1.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0, 0.0]})
    right = StatBin2d(breaks=([0, 1, 2], [-1, 1])).compute(df)
    assert right.sort_values("x")["count"].tolist() == [3, 1]  # (0,1] incl. 0, (1,2]
    left = StatBin2d(breaks=([0, 1, 2], [-1, 1]), closed="left").compute(df)
    assert left.sort_values("x")["count"].tolist() == [1, 3]  # [0,1), [1,2] incl. 2


def test_weight_drop_and_pairs():
    df = pd.DataFrame({"x": [0.5, 0.5, 2.5], "y": [0.5, 0.5, 0.5], "weight": [1.0, 2.0, 4.0]})
    out = StatBin2d(binwidth=(1, 1)).compute(df)
    assert sorted(out["count"]) == [3.0, 4.0]
    full = StatBin2d(binwidth=(1, 1), drop=False).compute(df)
    assert (full["count"] == 0).sum() == len(full) - 2
    out = StatBin2d(bins=(2, 1)).compute(df.drop(columns="weight"))
    assert out["count"].sum() == 3 and out["y"].nunique() == 1


def test_geom_builds_tiles():
    g = ggplot(cloud(), aes("x", "y")).add(geom_bin_2d(bins=10)).build()
    mob = g.layers[0]
    assert isinstance(stat_bin_2d().geom, GeomBin2d)
    assert len(mob.marks) == len(StatBin2d(bins=10).compute(cloud()))
    # fill mapped to the count: a continuous colour bar, distinct tile colours
    assert len(g.legends) == 1
    assert len({m.get_fill_color().to_hex() for m in mob.marks}) > 3
