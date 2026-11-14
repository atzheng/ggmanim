"""KDE (stats/_kde.py), stat_density and geom_density, against R/ggplot2
(fixtures from tests/fixtures/distributions_r.R)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ggmanim
from ggmanim import aes, after_stat, geom_density, ggplot, stat_density
from ggmanim.geoms import GeomArea, GeomDensity
from ggmanim.registry import COMPONENTS
from ggmanim.stats import StatDensity
from ggmanim.stats._kde import bandwidth, bw_nrd, bw_nrd0, compute_density, density

FIX = Path(__file__).parent / "fixtures"
DATA = pd.read_csv(FIX / "dist_data.csv")
REF = pd.read_csv(FIX / "dist_density_r.csv")


def test_registered_and_exported():
    for name in ("geom_density", "stat_density"):
        assert name in COMPONENTS and name in ggmanim.__all__
    assert isinstance(geom_density().geom, GeomDensity)
    assert isinstance(geom_density().stat, StatDensity)
    assert isinstance(stat_density().geom, GeomArea)


def test_bandwidths_match_r():
    ref = pd.read_csv(FIX / "dist_bw_r.csv")
    for row in ref.itertuples():
        x = DATA.loc[DATA.g == row.g, "v"].to_numpy()
        assert bw_nrd0(x) == pytest.approx(row.nrd0, rel=1e-10)
        assert bw_nrd(x) == pytest.approx(row.nrd, rel=1e-10)
        assert bandwidth(x, "silverman") == bandwidth(x, "nrd0")
        assert bandwidth(x, "scott") == bandwidth(x, "nrd")
    assert bandwidth([1.0, 2.0], 0.3) == 0.3
    assert bandwidth([1.0, 2.0, 4.0], lambda x: x.std()) == pytest.approx(np.std([1, 2, 4]))
    with pytest.raises(ValueError, match="unknown bandwidth"):
        bandwidth([1.0, 2.0], "sj")
    with pytest.raises(ValueError, match="positive"):
        bandwidth([1.0, 2.0], -1)


@pytest.mark.parametrize("case", ["default", "weighted", "bounded"])
def test_stat_density_matches_ggplot2(case):
    ref = REF[REF.case == case]
    if case == "default":
        p = ggplot(DATA, aes("v", color="g")).add(geom_density(n=64))
    elif case == "weighted":
        p = ggplot(DATA, aes("v", color="g", weight="w")).add(
            geom_density(n=64, adjust=0.7, kernel="epanechnikov")
        )
    else:
        p = ggplot(DATA[DATA.g == "b"], aes("v")).add(geom_density(n=64, bounds=(0, np.inf)))
    ld = p.build().layer_data[0]
    assert len(ld) == len(ref)
    for col in ("x", "density", "count", "scaled"):
        np.testing.assert_allclose(ld[col], ref[col], rtol=1e-8, atol=1e-12, err_msg=col)
    np.testing.assert_allclose(ld["y"], ld["density"])  # y = after_stat("density")


def test_density_integrates_to_one():
    rng = np.random.default_rng(3)
    x = rng.normal(size=200)
    for kernel in (
        "gaussian",
        "epanechnikov",
        "rectangular",
        "triangular",
        "biweight",
        "cosine",
        "optcosine",
    ):
        grid, y = density(x, bw=bw_nrd0(x), kernel=kernel)
        assert np.trapezoid(y, grid) == pytest.approx(1.0, abs=1e-2), kernel
    d = compute_density(np.abs(x), bounds=(0, np.inf))
    assert np.trapezoid(d["density"], d["x"]) == pytest.approx(1.0, abs=5e-3)
    with pytest.raises(ValueError, match="kernel"):
        density(x, bw=1.0, kernel="nope")


def test_after_stat_variables_trim_and_small_groups():
    df = pd.DataFrame({"v": [1.0, 2.0, 2.5, 4.0, 10.0, 11.0], "g": list("aaaabb")})
    ld = (
        ggplot(df, aes("v", fill="g"))
        .add(geom_density(aes(y=after_stat("count")), trim=True, n=32))
        .build()
        .layer_data[0]
    )
    a = ld[ld.group == 1]
    assert a["x"].min() == 1.0 and a["x"].max() == 4.0
    np.testing.assert_allclose(a["y"], a["density"] * 4)
    with pytest.warns(UserWarning, match="fewer than two"):
        df1 = pd.DataFrame({"v": [1.0, 2.0, 3.0, 9.0], "g": list("aaab")})
        ld = ggplot(df1, aes("v", fill="g")).add(geom_density()).build().layer_data[0]
    assert set(ld.group) == {1}


def test_geom_density_draws_and_orientation():
    p = ggplot(DATA, aes("v", fill="g")).add(geom_density(alpha=0.4)).build()
    marks = p.layers[0].marks
    assert len(marks) == 3
    pf = ggplot(DATA, aes(y="v")).add(geom_density(orientation="y")).build()
    ld = pf.layer_data[0]
    assert ld["flipped_aes"].all()
    np.testing.assert_allclose(ld["x"], ld["density"])
    assert ld["y"].min() == pytest.approx(DATA.v.min())
    assert len(pf.layers[0].marks) == 1
    assert GeomDensity().draw_key({}).width > 0


def test_stat_density_stacks():
    p = ggplot(DATA, aes("v", fill="g")).add(stat_density(n=64)).build()
    ld = p.layer_data[0]
    top = ld.groupby("x")["ymax"].max().to_numpy()
    total = ld.groupby("x")["density"].sum().to_numpy()
    np.testing.assert_allclose(top, total)
