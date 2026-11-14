"""stat_density_2d / geom_density_2d(_filled): kde2d vs R (MASS), contours vs ggplot2."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ggmanim import (
    aes,
    after_stat,
    geom_density_2d,
    geom_density_2d_filled,
    geom_raster,
    ggplot,
    stat_density_2d,
)
from ggmanim.geoms import GeomDensity2d, GeomDensity2dFilled
from ggmanim.registry import COMPONENTS
from ggmanim.stats import StatDensity2d, StatDensity2dFilled, bandwidth_nrd, kde2d

FIX = Path(__file__).parent / "fixtures"
DATA = pd.read_csv(FIX / "density2d_data.csv")
KDE_R = pd.read_csv(FIX / "kde2d_r.csv")
GG_R = pd.read_csv(FIX / "ggplot2d_r.csv")


def test_registered():
    for n in (
        "geom_density_2d",
        "geom_density_2d_filled",
        "geom_density2d",
        "stat_density_2d",
        "stat_density_2d_filled",
    ):
        assert n in COMPONENTS


def test_bandwidth_nrd_matches_r():
    assert bandwidth_nrd(DATA["x"]) == pytest.approx(KDE_R["hx"][0], rel=1e-10)
    assert bandwidth_nrd(DATA["y"]) == pytest.approx(KDE_R["hy"][0], rel=1e-10)


def test_kde2d_matches_mass():
    gx, gy, z = kde2d(DATA["x"], DATA["y"], n=20)
    assert np.allclose(np.tile(gx, 20), KDE_R["x"], atol=1e-9)
    assert np.allclose(np.repeat(gy, 20), KDE_R["y"], atol=1e-9)
    assert np.allclose(z.ravel(order="F"), KDE_R["density"], rtol=1e-9, atol=1e-14)


def test_kde2d_bad_bandwidth():
    with pytest.raises(ValueError, match="strictly positive"):
        kde2d([0, 1], [0, 1], h=0)


def test_contours_match_ggplot2():
    # same levels and the same number of contour vertices per level as ggplot2 4.0
    out = StatDensity2d().compute(DATA[["x", "y"]])
    ours = out.groupby("level").size()
    ref = GG_R[GG_R["kind"] == "density_level"]
    assert np.allclose(ours.index.to_numpy(), ref["a"].astype(float))
    assert ours.to_numpy().tolist() == ref["c"].astype(int).tolist()


def test_grid_without_contour():
    out = StatDensity2d(n=30, contour=False).compute(DATA[["x", "y"]])
    assert len(out) == 900
    assert {"density", "ndensity", "count", "n"} <= set(out.columns)
    assert out["ndensity"].max() == pytest.approx(1.0)
    assert np.allclose(out["count"], 60 * out["density"])
    # the grid spans the data range; density integrates to about 1 inside it
    dx = np.diff(np.unique(out["x"]))[0]
    dy = np.diff(np.unique(out["y"]))[0]
    assert 0.8 < out["density"].sum() * dx * dy < 1.05


def test_h_and_adjust():
    a = StatDensity2d(n=20, contour=False, h=(1.0, 2.0)).compute(DATA[["x", "y"]])
    b = StatDensity2d(n=20, contour=False, h=(0.5, 1.0), adjust=2).compute(DATA[["x", "y"]])
    assert np.allclose(a["density"], b["density"])


def bivariate(n=2000, seed=3):
    rng = np.random.default_rng(seed)
    xy = rng.multivariate_normal([0, 0], [[1.0, 0.7], [0.7, 1.0]], size=n)
    return pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]})


def _inside(px, py, ring):
    x, y = ring[:, 0], ring[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    crosses = (y > py) != (y2 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x + (py - y) * (x2 - x) / (y2 - y)
    return bool(np.count_nonzero(crosses & (px < xi)) % 2)


def test_bivariate_normal_gives_nested_ellipses():
    out = StatDensity2d(bins=6).compute(bivariate())
    rings = []
    for lev, gdf in out.groupby("level"):
        pieces = gdf.groupby("group")
        if len(pieces) != 1:  # a tail level may break into blobs; only inner ones count
            continue
        ring = gdf[["x", "y"]].to_numpy()
        assert np.allclose(ring[0], ring[-1])  # closed
        rings.append((lev, ring))
    assert len(rings) >= 3
    for (_, outer), (_, inner) in zip(rings, rings[1:]):
        assert all(_inside(px, py, outer) for px, py in inner[:-1])
    # elongated along the correlation: an ellipse tilted at 45 degrees
    _, ring = rings[len(rings) // 2]
    centred = ring - ring.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(centred.T))
    major = evecs[:, -1]
    assert abs(abs(major[0]) - abs(major[1])) < 0.1
    assert evals[-1] / evals[0] > 3


def test_filled_bands():
    out = StatDensity2dFilled(bins=5).compute(bivariate(500))
    assert isinstance(out["level"].dtype, pd.CategoricalDtype)
    assert out["level"].cat.ordered
    assert {"subgroup", "level_low", "level_high", "nlevel"} <= set(out.columns)
    # bands are contiguous: each band's high edge is the next one's low edge
    lows = sorted(out["level_low"].unique())
    highs = sorted(out["level_high"].unique())
    assert lows[1:] == highs[:-1]


def test_geoms_build():
    g = (
        ggplot(bivariate(300), aes("x", "y"))
        .add(geom_density_2d_filled(), geom_density_2d(color="white"))
        .build()
    )
    bands, lines = g.layers[0], g.layers[1]
    assert isinstance(GeomDensity2dFilled(), GeomDensity2dFilled)
    assert len(bands.marks) >= 5 and len(lines.marks) >= 5
    # fill mapped to the band label: a discrete legend with one key per band
    assert len(g.legends) == 1
    colours = {m.get_fill_color().to_hex() for m in bands.marks}
    assert len(colours) == len(bands.marks)


def test_contour_var_and_raster():
    g = (
        ggplot(bivariate(300), aes("x", "y"))
        .add(geom_raster(aes(fill=after_stat("density")), stat="density_2d", contour=False, n=30))
        .build()
    )
    assert len(g.layers[0].marks) > 1
    out = StatDensity2d(contour_var="ndensity", bins=4).compute(bivariate(300))
    assert out["level"].max() <= 1.0
    with pytest.raises(ValueError, match="contour_var"):
        StatDensity2d(contour_var="nope").compute(bivariate(50))


def test_groups_share_levels():
    df = pd.concat(
        [bivariate(300).assign(g="a"), (bivariate(300, seed=4) + 3).assign(g="b")],
        ignore_index=True,
    )
    out = StatDensity2d(breaks=[0.02, 0.05, 0.1]).compute(df.rename(columns={"g": "color"}))
    assert set(out["color"]) == {"a", "b"}
    assert set(out["level"]) <= {0.02, 0.05, 0.1}
    assert out.groupby("group")["color"].nunique().max() == 1


def test_stat_layer_default_geom():
    layer = stat_density_2d()
    assert isinstance(layer.geom, GeomDensity2d)
