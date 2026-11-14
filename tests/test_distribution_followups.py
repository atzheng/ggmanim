"""Distribution follow-ups (ggm-8ju): positions that flip horizontal layers
themselves, the public interval helpers, bw = "ucv"/"bcv"/"SJ", weighted boxplot
quantiles, plot-level horizontal densities and faceted boxplots/violins.

Fixtures: tests/fixtures/bw_rules_r.R and tests/fixtures/flipped_positions_r.R
(R 4.6.0, ggplot2 4.0.3).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from manim import VGroup

import ggmanim.geoms.boxplot as boxplot_module
from ggmanim import (
    aes,
    after_stat,
    facet_wrap,
    geom_boxplot,
    geom_density,
    geom_violin,
    ggplot,
    position_dodge,
    position_dodge2,
    position_jitterdodge,
    stat_density,
)
from ggmanim.geoms import DensityLayer, clipped_segments, flip_panel, style
from ggmanim.positions import PositionDodge2
from ggmanim.stats import bw_bcv, bw_sj, bw_ucv, resolve_bw
from ggmanim.stats.boxplot import flip_names, weighted_quantile

FIX = Path(__file__).parent / "fixtures"
DATA = pd.read_csv(FIX / "dist_data.csv")
DATA_H = DATA.assign(h=np.where(np.arange(1, len(DATA) + 1) % 2 == 0, "q", "p"))


def _ld(plot, i=0):
    return plot.build().layer_data[i]


def _first_per_group(ld):
    return ld.sort_values("group", kind="stable").drop_duplicates("group")


# -- 1. positions flip horizontal layers themselves -----------------------------
FLIPPED = pd.read_csv(FIX / "flipped_positions_r.csv")


def _ref(case):
    return FLIPPED[FLIPPED["case"] == case].reset_index(drop=True)


def test_orientedposition_retired():
    assert not hasattr(boxplot_module, "OrientedPosition")
    assert not hasattr(boxplot_module, "oriented")
    assert isinstance(geom_boxplot().position, PositionDodge2)


def test_horizontal_boxplot_dodge2_matches_ggplot2():
    ld = _first_per_group(_ld(ggplot(DATA_H, aes("v", "g", fill="h")).add(geom_boxplot())))
    ref = _ref("boxplot_dodge2")
    cols = ["y", "ymin", "ymax", "xmin", "xmax"]
    np.testing.assert_allclose(ld[cols].to_numpy(float), ref[cols].to_numpy(float), rtol=1e-9)


@pytest.mark.parametrize("geom", [geom_boxplot, geom_violin])
def test_horizontal_stack_is_flipped_once(geom):
    """A horizontal boxplot/violin with position="stack" used to be flipped twice
    (once by the OrientedPosition wrapper, once by position_stack itself)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hor = _ld(ggplot(DATA, aes("v", "g")).add(geom(position="stack")))
        ver = _ld(ggplot(DATA, aes("g", "v")).add(geom(position="stack")))
    assert hor["flipped_aes"].all() and not ver["flipped_aes"].any()
    flipped = flip_names(hor)
    cols = ["x", "xmin", "xmax", "ymin", "ymax"] if geom is geom_boxplot else ["x", "xmin", "y"]
    for col in cols:
        np.testing.assert_allclose(flipped[col].to_numpy(float), ver[col].to_numpy(float))
    if geom is geom_boxplot:
        ref = _ref("boxplot_stack")
        cols = ["y", "ymin", "ymax", "xmin", "xmax"]
        np.testing.assert_allclose(hor[cols].to_numpy(float), ref[cols].to_numpy(float))


def test_horizontal_violin_dodge_matches_ggplot2():
    ld = _first_per_group(_ld(ggplot(DATA_H, aes("v", "g", fill="h")).add(geom_violin())))
    ref = _ref("violin_dodge")
    np.testing.assert_allclose(
        ld[["y", "ymin", "ymax"]].to_numpy(float), ref[["y", "ymin", "ymax"]].to_numpy(float)
    )


def test_horizontal_jitterdodge_matches_ggplot2():
    """Unit level (points carry no ``flipped_aes`` column, and positions cannot
    yet guess the orientation from a discrete y scale): the layer data of
    ``aes(v, g, fill = h)`` with ``flipped_aes`` set dodges along y as in R."""
    codes = DATA_H["g"].map({"a": 1, "b": 2, "c": 3}).to_numpy(float)
    group = (codes - 1) * 2 + (DATA_H["h"] == "q").to_numpy() + 1
    df = pd.DataFrame(
        {
            "x": DATA_H["v"].to_numpy(float),
            "y": codes,
            "fill": DATA_H["h"],
            "group": group.astype(int),
            "PANEL": 1,
            "flipped_aes": True,
        }
    )
    pos = position_jitterdodge(jitter_width=0)
    params = pos.setup_params(df)
    assert params["flipped_aes"] and params["ndodge"] == 2
    out = pos.compute_layer(pos.setup_data(df, params), params, {})
    np.testing.assert_allclose(out["x"], df["x"])  # values untouched (no jitter)
    ref = _ref("jitterdodge")
    mine = sorted(zip(out["group"], out["y"].round(9), out["x"].round(9)))
    theirs = sorted(zip(ref["group"], ref["y"].round(9), ref["xmin"].round(9)))
    np.testing.assert_allclose(np.asarray(mine, float), np.asarray(theirs, float), atol=1e-8)
    # and the vertical twin dodges along x by the same amounts
    vert = flip_names(df).assign(flipped_aes=False)
    params = pos.setup_params(vert)
    vout = pos.compute_layer(pos.setup_data(vert, params), params, {})
    np.testing.assert_allclose(vout["x"], out["y"])


@pytest.mark.parametrize("pos", [position_dodge(width=0.6), position_dodge2(padding=0.2)])
def test_dodge_flips_on_flipped_aes(pos):
    """Unit level: a flipped layer is dodged exactly like its transposed twin."""
    vert = pd.DataFrame(
        {
            "x": [1.0, 1.0, 2.0],
            "xmin": [0.6, 0.6, 1.6],
            "xmax": [1.4, 1.4, 2.4],
            "ymin": [0.0, 0.0, 0.0],
            "ymax": [3.0, 2.0, 5.0],
            "group": [1, 2, 3],
            "PANEL": 1,
            "flipped_aes": False,
        }
    )
    hor = flip_names(vert).assign(flipped_aes=True)

    def run(df):
        params = pos.setup_params(df)
        return pos.compute_layer(pos.setup_data(df, params), params, {})

    a, b = run(vert), flip_names(run(hor))
    for col in ("x", "xmin", "xmax", "ymin", "ymax"):
        np.testing.assert_allclose(b[col].to_numpy(float), a[col].to_numpy(float))
    assert not np.allclose(a["xmin"], vert["xmin"])  # something was dodged


# -- 2. public interval helpers ------------------------------------------------
def test_interval_helpers_are_public():
    import ggmanim.geoms.interval as interval

    for name in ("clipped_segments", "style", "flip_panel"):
        assert name in interval.__all__
        assert not hasattr(interval, "_" + name)

    class Coord:
        clip = "on"

        def transform(self, x, y):
            x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
            return np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])

    coord, pp = flip_panel(Coord(), None)
    assert pp is None
    np.testing.assert_allclose(coord.transform([1.0], [2.0])[0], [2.0, 1.0, 0.0])
    pieces = clipped_segments(Coord(), None, [0, 1], [0, 0], [1, 1], [1, np.nan])
    assert len(pieces) == 1 and pieces[0].shape == (2, 3)
    from ggmanim.geoms.base import multi_path_mobject

    mob = style(multi_path_mobject(pieces, None, 2.0, 1.0), {"linetype": "dashed"}, "#000", 2.0)
    assert isinstance(mob, VGroup) or mob.get_stroke_width() > 0


# -- 3. bandwidth rules --------------------------------------------------------
def _samples():
    out = {f"dist_{g}": grp["v"].to_numpy(float) for g, grp in DATA.groupby("g")}
    big = pd.read_csv(FIX / "bw_rules_data.csv")
    out.update({c: grp["v"].to_numpy(float) for c, grp in big.groupby("case")})
    return out


@pytest.mark.parametrize("row", pd.read_csv(FIX / "bw_rules_r.csv").to_dict("records"))
def test_bw_rules_match_r(row):
    x = _samples()[row["case"]]
    assert len(x) == row["n"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "minimum occurred at one end of the range"
        got = [bw_ucv(x), bw_bcv(x), bw_sj(x), bw_sj(x, method="dpi")]
    want = [row["ucv"], row["bcv"], row["sj"], row["sj_dpi"]]
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_resolve_bw_names_and_errors():
    x = DATA["v"].to_numpy(float)
    assert resolve_bw(x, "SJ") == resolve_bw(x, "sj-ste") == bw_sj(x)
    assert resolve_bw(x, "sj-dpi") == bw_sj(x, method="dpi")
    assert resolve_bw(x, "nrd0") == "nrd0" and resolve_bw(x, 0.3) == 0.3
    with pytest.raises(ValueError):
        bw_ucv([1.0])
    with pytest.raises(ValueError):
        bw_sj(x, method="nope")
    with pytest.raises(ValueError):
        bw_bcv(np.ones(5))


def test_bw_rules_end_to_end_match_ggplot2():
    ref = pd.read_csv(FIX / "bw_rules_ggplot_r.csv")
    dens = ref[ref["case"] == "density_SJ"]
    ld = _ld(ggplot(DATA, aes("v", colour="g")).add(geom_density(bw="SJ", n=32)))
    np.testing.assert_allclose(ld["x"], dens["x"], rtol=1e-9)
    np.testing.assert_allclose(ld["density"], dens["y"], rtol=1e-7, atol=1e-12)
    vio = ref[ref["case"] == "violin_bcv"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ld = _ld(ggplot(DATA, aes("g", "v")).add(geom_violin(bw="bcv")))
    ld = ld[ld["quantile"].isna()].reset_index(drop=True)
    ld = ld[np.arange(1, len(ld) + 1) % 32 == 1]
    np.testing.assert_allclose(ld["y"], vio["x"], rtol=1e-9)
    np.testing.assert_allclose(ld["density"], vio["y"], rtol=1e-7)


# -- 4. weighted boxplot quantiles ---------------------------------------------
def _check_loss(y, w, tau, b):
    u = y - b
    return float(np.sum(w * u * (tau - (u < 0))))


@pytest.mark.parametrize("seed", range(6))
def test_weighted_quantiles_minimise_rq_objective(seed):
    """ggplot2 computes weighted boxplot stats with ``quantreg::rq(y ~ 1,
    weights = w, tau = qs)``, i.e. the minimiser of the weighted check loss.
    quantreg is not installed in the reference R (4.6.0), so instead of an R
    fixture this checks that :func:`weighted_quantile` attains the minimum of
    that objective (a piecewise-linear convex function, so the minimum is at a
    data value). When the minimiser is not unique (cumulative weight exactly
    ``tau``) rq's simplex may return either end; ggmanim returns the lower one."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=int(rng.integers(5, 40)))
    w = rng.uniform(0.1, 3.0, len(y))
    taus = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    got = weighted_quantile(y, w, taus)
    for tau, b in zip(taus, got):
        best = min(_check_loss(y, w, tau, c) for c in y)
        assert _check_loss(y, w, tau, b) <= best + 1e-12 * max(1.0, abs(best))
    lo_hi = weighted_quantile(y, w, [0.0, 1.0])
    assert lo_hi[0] == y.min() and lo_hi[1] == y.max()


# -- 5. horizontal densities from the plot mapping -----------------------------
@pytest.mark.parametrize("ctor", [geom_density, stat_density])
def test_density_orientation_from_plot_mapping(ctor):
    layer = ctor()
    assert isinstance(layer, DensityLayer)
    hor = _ld(ggplot(DATA, aes(y="v")).add(layer))
    assert hor["flipped_aes"].all()
    ver = _ld(ggplot(DATA, aes(x="v")).add(ctor()))
    assert not ver["flipped_aes"].any()
    np.testing.assert_allclose(hor["y"], ver["x"])
    np.testing.assert_allclose(hor["x"], ver["y"])
    # a layer-level x wins over the plot-level y
    lay = _ld(ggplot(DATA, aes(y="v")).add(ctor(aes(x="v", y=after_stat("density")))))
    assert not lay["flipped_aes"].any()
    # orientation= still forces it
    forced = _ld(ggplot(DATA, aes(y="v")).add(ctor(orientation="y")))
    np.testing.assert_allclose(forced["x"], hor["x"])
    # an explicit after_stat y is not mistaken for a horizontal density
    cnt = _ld(ggplot(DATA, aes(x="v")).add(ctor(aes(y=after_stat("count")))))
    assert not cnt["flipped_aes"].any()


def test_density_layer_survives_replace():
    layer = geom_density().replace(name="d")
    assert isinstance(layer, DensityLayer) and layer.name == "d"
    built = ggplot(DATA, aes(y="v")).add(layer).build()
    assert built.layer_data["d"]["flipped_aes"].all()


# -- 6. boxplots and violins with facet_wrap -----------------------------------
@pytest.mark.parametrize("geom", [geom_boxplot, geom_violin])
@pytest.mark.parametrize("horizontal", [False, True])
def test_boxplot_violin_facet_wrap(geom, horizontal):
    mapping = aes("v", "h", fill="h") if horizontal else aes("h", "v", fill="h")
    plot = ggplot(DATA_H, mapping).add(geom(), facet_wrap("g"))
    built = plot.build()
    ld = built.layer_data[0]
    assert sorted(ld["PANEL"].unique()) == [1, 2, 3]
    assert ld["flipped_aes"].all() == horizontal
    # every panel matches the same plot on that panel's rows alone
    for panel, level in zip((1, 2, 3), sorted(DATA_H["g"].unique())):
        alone = _ld(ggplot(DATA_H[DATA_H["g"] == level], mapping).add(geom()))
        mine = ld[ld["PANEL"] == panel].reset_index(drop=True)
        cols = [
            c
            for c in alone.columns
            if c in mine.columns
            and c not in ("PANEL", "group")
            and pd.api.types.is_numeric_dtype(alone[c])
        ]
        assert {"ymin", "ymax"} <= set(cols) if horizontal else {"xmin", "xmax"} <= set(cols)
        np.testing.assert_allclose(
            mine[cols].to_numpy(float), alone[cols].to_numpy(float), rtol=1e-9
        )
    assert len(built.layers[0]) > 0
