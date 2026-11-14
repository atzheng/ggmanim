"""stat_ydensity / geom_violin (fixtures from tests/fixtures/distributions_r.R)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ggmanim
from ggmanim import aes, geom_violin, ggplot, stat_ydensity
from ggmanim.geoms import GeomViolin
from ggmanim.registry import COMPONENTS
from ggmanim.stats import StatYdensity

FIX = Path(__file__).parent / "fixtures"
DATA = pd.read_csv(FIX / "dist_data.csv")
REF = pd.read_csv(FIX / "dist_violin_r.csv")


def test_registered_and_exported():
    for name in ("geom_violin", "stat_ydensity"):
        assert name in COMPONENTS and name in ggmanim.__all__
    assert isinstance(geom_violin().geom, GeomViolin)
    assert isinstance(geom_violin().stat, StatYdensity)
    assert isinstance(stat_ydensity().geom, GeomViolin)


def _subset(ld: pd.DataFrame) -> pd.DataFrame:
    """The rows written by the R script: the quantile rows (ggplot2 4.0 appends
    them to each group's grid) and every 16th row of the layer data."""
    idx = np.arange(1, len(ld) + 1)
    return ld[ld["quantile"].notna().to_numpy() | (idx % 16 == 1)]


@pytest.mark.parametrize("case", ["area", "count"])
def test_ydensity_matches_ggplot2(case):
    ref = REF[REF.case == case]
    kw = {} if case == "area" else {"scale": "count", "trim": False}
    ld = ggplot(DATA, aes("g", "v")).add(geom_violin(**kw)).build().layer_data[0]
    got = _subset(ld)
    assert len(got) == len(ref)
    for col in ("x", "y", "density", "violinwidth", "xmin", "xmax"):
        np.testing.assert_allclose(got[col], ref[col], rtol=1e-8, atol=1e-12, err_msg=col)
    np.testing.assert_allclose(got["quantile"], ref["quantile"])


def test_violin_mirrors_and_widths():
    p = ggplot(DATA, aes("g", "v")).add(geom_violin(scale="width")).build()
    marks = p.layers[0].marks
    assert len(marks) == 3
    for i, m in enumerate(marks):
        pts = m.get_all_points()
        cx = p.panel.transform(np.array([i + 1.0]), np.array([0.0]))[0, 0]
        # symmetric about x
        assert pts[:, 0].min() + pts[:, 0].max() == pytest.approx(2 * cx, abs=1e-6)
    ld = p.layer_data[0]
    assert ld.groupby("group")["violinwidth"].max().to_numpy() == pytest.approx(1.0)


def test_dodged_violins_and_quantile_lines():
    df = DATA.assign(s=np.where(np.arange(len(DATA)) % 2, "m", "f"))
    p = ggplot(df, aes("g", "v", fill="s")).add(geom_violin(draw_quantiles=[0.5])).build()
    ld = p.layer_data[0]
    xs = ld.groupby("group")["x"].first().to_numpy()
    np.testing.assert_allclose(xs, [0.775, 1.225, 1.775, 2.225, 2.775, 3.225])
    assert sorted(ld["quantile"].dropna().unique()) == [0.5]
    marks = p.layers[0].marks
    assert len(marks) == 6 and all(len(m) == 2 for m in marks)  # polygon + quantile line


def test_horizontal_violin_and_key():
    p = ggplot(DATA, aes("v", "g")).add(geom_violin()).build()
    ld = p.layer_data[0]
    assert ld["flipped_aes"].all() and {"ymin", "ymax", "violinwidth"} <= set(ld.columns)
    assert len(p.layers[0].marks) == 3
    assert GeomViolin().draw_key({"fill": "#00ff00"}).width > 0


def test_small_groups_dropped():
    df = pd.DataFrame({"g": list("aaab"), "v": [1.0, 2.0, 3.0, 4.0]})
    with pytest.warns(UserWarning, match="fewer than two"):
        ld = ggplot(df, aes("g", "v")).add(geom_violin()).build().layer_data[0]
    assert set(ld["x"]) == {1.0}
