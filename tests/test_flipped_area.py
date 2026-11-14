"""Horizontal (``flipped_aes``) areas and ribbons: geom_area / geom_ribbon and
position_stack / position_fill flip, compute and flip back as in ggplot2.

The references in ``fixtures/flipped_area_r.csv`` are ggplot2's ``layer_data`` for
``geom_area(aes(x = v, y = t, fill = g), orientation = "y")`` and friends
(``fixtures/flipped_area_r.R``)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ggmanim import aes, geom_area, geom_density, geom_ribbon, ggplot, position_stack
from ggmanim.geoms import GeomArea, GeomRibbon
from ggmanim.positions import PositionFill, PositionStack
from ggmanim.theme import Theme

FIX = Path(__file__).parent / "fixtures"
REF = pd.read_csv(FIX / "flipped_area_r.csv")
COLS = ["x", "xmin", "xmax", "y"]


def _data(name: str) -> pd.DataFrame:
    return pd.read_csv(FIX / f"flipped_area_{name}.csv")


def _layer(df: pd.DataFrame, **params) -> pd.DataFrame:
    p = ggplot(df, aes(x="v", y="t", fill="g")).add(geom_area(**params), name="a").build()
    return p.layer_data["a"]


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["g", "y", "x"], kind="stable").reset_index(drop=True)


def _with_g(ld: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    levels = sorted(df["g"].unique())
    groups = sorted(ld["group"].unique())
    return ld.assign(g=ld["group"].map(dict(zip(groups, levels))))


@pytest.mark.parametrize(
    "case, name, params",
    [
        ("align", "data", {"orientation": "y"}),
        ("identity", "shared", {"stat": "identity", "orientation": "y"}),
        ("fill", "shared", {"stat": "identity", "position": "fill", "orientation": "y"}),
        (
            "reverse",
            "shared",
            {"stat": "identity", "position": position_stack(reverse=True), "orientation": "y"},
        ),
    ],
)
def test_matches_ggplot2(case, name, params):
    df = _data(name)
    ld = _with_g(_layer(df, **params), df)
    ref = _sorted(REF[REF["case"] == case])
    ours = _sorted(ld)
    assert len(ours) == len(ref)
    assert list(ours["g"]) == list(ref["g"])
    for c in COLS:
        np.testing.assert_allclose(ours[c].to_numpy(float), ref[c].to_numpy(float), atol=1e-8)
    assert ours["flipped_aes"].all()
    assert "ymin" not in ours and "ymax" not in ours


def test_flipped_is_transpose_of_vertical():
    df = _data("data")
    flat = ggplot(df, aes(x="t", y="v", fill="g")).add(geom_area(), name="a").build()
    vert = flat.layer_data["a"]
    horiz = _layer(df, orientation="y")
    for a, b in (("x", "y"), ("y", "x"), ("ymin", "xmin"), ("ymax", "xmax")):
        np.testing.assert_allclose(vert[a].to_numpy(float), horiz[b].to_numpy(float))
    assert not vert["flipped_aes"].any()


def test_position_stack_flipped_directly():
    df = pd.DataFrame(
        {
            "y": [0.0, 0.0, 1.0, 1.0],
            "x": [1.0, 2.0, 3.0, -1.0],
            "xmin": 0.0,
            "xmax": [1.0, 2.0, 3.0, -1.0],
            "group": [1, 2, 1, 2],
            "flipped_aes": True,
        }
    )
    pos = PositionStack()
    params = pos.setup_params(df)
    assert params["flipped_aes"] and params["var"] == "ymax"
    out = pos.compute_layer(pos.setup_data(df, params), params)
    # group 2 at the bottom (first group on top), negatives stacked separately
    np.testing.assert_allclose(out["xmin"], [2.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(out["xmax"], [3.0, 2.0, 3.0, -1.0])  # xmax < xmin kept
    np.testing.assert_allclose(out["y"], df["y"])
    assert list(out.columns) == list(df.columns)
    fill = PositionFill()
    fp = fill.setup_params(df)
    out = fill.compute_layer(fill.setup_data(df, fp), fp)
    np.testing.assert_allclose(out["xmax"], [1.0, 2 / 3, 1.0, -1.0])


def test_unflipped_stack_unchanged_by_false_flag():
    df = pd.DataFrame({"x": [1.0, 1.0], "y": [1.0, 2.0], "group": [1, 2]})
    pos = PositionStack()
    pa = pos.setup_params(df)
    a = pos.compute_layer(pos.setup_data(df, pa), pa)
    b_in = df.assign(flipped_aes=False)
    pb = pos.setup_params(b_in)
    b = pos.compute_layer(pos.setup_data(b_in, pb), pb)
    pd.testing.assert_frame_equal(a, b.drop(columns="flipped_aes"))


def test_is_flipped_rules():
    g = GeomRibbon()
    assert g.is_flipped(pd.DataFrame({"y": [0], "xmin": [0], "xmax": [1]}))
    assert not g.is_flipped(pd.DataFrame({"x": [0], "ymin": [0], "ymax": [1]}))
    assert g.is_flipped(pd.DataFrame({"x": [0], "ymin": [0]}), {"orientation": "y"})
    # a flipped_aes column written by the stat wins over orientation (ggplot2)
    assert g.is_flipped(pd.DataFrame({"x": [0], "flipped_aes": [True]}), {"orientation": "x"})
    assert not GeomArea().is_flipped(pd.DataFrame({"x": [0], "y": [1]}))
    with pytest.raises(ValueError, match="orientation"):
        geom_area(orientation="z")
    with pytest.raises(ValueError, match="orientation"):
        geom_ribbon(orientation="z")


def test_horizontal_ribbon_draws_along_y():
    df = pd.DataFrame({"t": [0.0, 1.0, 2.0], "lo": [1.0, 2.0, 1.0], "hi": [2.0, 4.0, 3.0]})
    p = ggplot(df, aes(y="t", xmin="lo", xmax="hi")).add(geom_ribbon(), name="r").build()
    assert p.layer_data["r"]["flipped_aes"].all()
    pts = p.layers["r"][0].get_all_points()
    corner_lo = p.panel.transform(1.0, 0.0)[0]
    corner_hi = p.panel.transform(4.0, 2.0)[0]
    np.testing.assert_allclose(pts.min(axis=0)[:2], corner_lo[:2], atol=1e-6)
    np.testing.assert_allclose(pts.max(axis=0)[:2], corner_hi[:2], atol=1e-6)
    # the band at t = 1 spans lo..hi = 2..4 horizontally
    mid = pts[np.isclose(pts[:, 1], p.panel.transform(0.0, 1.0)[0][1])]
    xs = sorted({round(v, 6) for v in mid[:, 0]})
    assert xs[0] == pytest.approx(p.panel.transform(2.0, 1.0)[0][0], abs=1e-5)
    assert xs[-1] == pytest.approx(p.panel.transform(4.0, 1.0)[0][0], abs=1e-5)


def test_horizontal_area_marks_keep_original_names():
    df = _data("shared")
    p = ggplot(df, aes(x="v", y="t", fill="g")).add(
        geom_area(stat="identity", orientation="y"), name="a"
    )
    plot = p.build()
    assert len(plot.layers["a"]) == 3
    pts = np.concatenate([m.get_all_points() for m in plot.layers["a"]])
    # total stack width at t = 2 is 3 + 1 + 4 = 8 (data), all marks inside the panel
    right = plot.panel.transform(8.0, 2.0)[0][0]
    assert pts[:, 0].max() == pytest.approx(right, abs=1e-6)


def test_horizontal_density_still_flipped():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"v": rng.normal(size=60)})
    ld = ggplot(df, aes(y="v")).add(geom_density(orientation="y"), name="d").build().layer_data["d"]
    np.testing.assert_allclose(ld["xmin"], 0.0)
    np.testing.assert_allclose(ld["xmax"], ld["density"])
    assert "ymax" not in ld


def test_stacked_horizontal_densities():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"v": rng.normal(size=80), "g": ["a", "b"] * 40})
    p = ggplot(df, aes(y="v", fill="g")).add(
        geom_density(orientation="y", position="stack"), name="d"
    )
    ld = p.build().layer_data["d"]
    top = ld[ld["group"] == ld["group"].min()]
    bottom = ld[ld["group"] == ld["group"].max()]
    np.testing.assert_allclose(bottom["xmin"], 0.0)
    np.testing.assert_allclose(top["xmin"].to_numpy(), bottom["xmax"].to_numpy())


def test_theme_validates_facets_in_post_init():
    with pytest.raises(ValueError, match="strip_placement"):
        Theme(strip_placement="sideways")
