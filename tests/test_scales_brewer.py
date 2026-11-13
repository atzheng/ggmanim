"""ColorBrewer scales: brewer (discrete), distiller (continuous), fermenter (binned).

Reference values from R 4 / ggplot2 / scales / RColorBrewer 1.1-3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ggmanim
from ggmanim import aes, ggplot
from ggmanim._brewer import BREWER, BREWER_TYPES, brewer_max_n
from ggmanim.scales.brewer import (
    ScaleBinnedColour,
    brewer_pal,
    brewer_palette_name,
    scale_color_brewer,
    scale_color_distiller,
    scale_color_fermenter,
    scale_colour_brewer,
    scale_fill_brewer,
    scale_fill_distiller,
)
from ggmanim.scales.colour import ScaleContinuousColour
from ggmanim.scales.discrete import ScaleDiscrete


def hexes(cols) -> list[str]:
    return [c.to_hex().upper() for c in cols]


# --------------------------------------------------------------- tables
def test_tables_complete():
    assert sum(len(v) for v in BREWER_TYPES.values()) == 35
    assert set(BREWER) == {n for v in BREWER_TYPES.values() for n in v}
    for name, table in BREWER.items():
        assert min(table) == 3
        assert sorted(table) == list(range(3, brewer_max_n(name) + 1))
        for n, cols in table.items():
            assert len(cols) == n
            assert all(c.startswith("#") and len(c) == 7 for c in cols)
    assert brewer_max_n("Set1") == 9
    assert brewer_max_n("Paired") == 12
    assert brewer_max_n("Blues") == 9
    assert brewer_max_n("RdBu") == 11


def test_per_n_schemes_are_not_truncations():
    # ColorBrewer designs each n separately: Blues 3 is not the first 3 of Blues 9
    assert BREWER["Blues"][3] == ("#DEEBF7", "#9ECAE1", "#3182BD")
    assert BREWER["Blues"][9][:3] != BREWER["Blues"][3]


# --------------------------------------------------------------- lookup
def test_lookup_by_name_case_insensitive():
    assert brewer_palette_name("Set1") == "Set1"
    assert brewer_palette_name("set1") == "Set1"
    assert brewer_palette_name("rdylbu", type="qual") == "RdYlBu"  # name wins over type


def test_lookup_by_index_and_type():
    assert brewer_palette_name(1) == "Blues"  # default type="seq"
    assert brewer_palette_name(1, "div") == "BrBG"
    assert brewer_palette_name(2, "qual") == "Dark2"
    assert brewer_palette_name(18, "seq") == "YlOrRd"
    assert brewer_palette_name(1, "qualitative") == "Accent"
    # scales::brewer_pal("qual", 2)(3)
    assert hexes(brewer_pal("qual", 2)(3)) == ["#1B9E77", "#D95F02", "#7570B3"]


def test_lookup_errors():
    with pytest.raises(ValueError, match="Unknown ColorBrewer palette"):
        brewer_palette_name("Nope")
    with pytest.raises(ValueError, match="out of range"):
        brewer_palette_name(10, "div")
    with pytest.raises(ValueError, match="type must be"):
        brewer_palette_name(1, "fancy")
    with pytest.raises(ValueError, match="direction"):
        brewer_pal(direction=0)


def test_small_n_uses_three_class_scheme():
    # scales::brewer_pal(palette = "Set1")(2)
    assert hexes(brewer_pal(palette="Set1")(2)) == ["#E41A1C", "#377EB8"]
    assert hexes(brewer_pal(palette="Blues")(1)) == ["#DEEBF7"]
    assert brewer_pal()(0) == []


def test_direction_reverses():
    fwd = hexes(brewer_pal("seq", "Greens")(5))
    rev = hexes(brewer_pal("seq", "Greens", -1)(5))
    assert fwd == list(BREWER["Greens"][5])
    assert rev == fwd[::-1]


def test_too_many_colours_helpful_error():
    with pytest.raises(ValueError, match=r"'Set1' has at most 9 colours, 10 needed.*cycle=True"):
        brewer_pal("qual", "Set1")(10)
    with pytest.raises(ValueError, match="distiller"):
        brewer_pal("seq", "Blues")(10)
    # non-strict returns the largest scheme for the caller to recycle
    assert len(brewer_pal("qual", "Set1", strict=False)(12)) == 9


# --------------------------------------------------------------- brewer scale
def test_scale_brewer_maps_levels():
    s = scale_color_brewer(palette="Set1")
    assert isinstance(s, ScaleDiscrete)
    s.train(pd.Series(["a", "b", "c", "a"]))
    assert hexes(s.map(["a", "b", "c"])) == ["#E41A1C", "#377EB8", "#4DAF4A"]
    assert s.aesthetics == ("color",)
    assert scale_fill_brewer().aesthetics == ("fill",)
    assert scale_colour_brewer().aesthetics == ("color",)


def test_scale_brewer_too_many_levels_and_cycle():
    df = pd.DataFrame({"x": range(10), "y": range(10), "g": list("abcdefghij")})
    p = ggplot(df, aes("x", "y", color="g")).geom_point().scale_color_brewer(palette="Set1")
    with pytest.raises(ValueError, match="at most 9 colours"):
        p.build()
    s = scale_color_brewer(palette="Set1", cycle=True)
    s.train(df["g"])
    cols = hexes(s.map(list("abcdefghij")))
    assert cols[9] == cols[0] == "#E41A1C"


# --------------------------------------------------------------- distiller
def test_distiller_endpoints_default_direction():
    # ggplot2: scale_colour_distiller(palette = "Blues") on y = 1:10 -> "#084594" ... "#EFF3FF"
    s = scale_color_distiller(palette="Blues")
    assert isinstance(s, ScaleContinuousColour)
    s.train(np.arange(1, 11, dtype=float))
    lo, hi = hexes(s.map(np.array([1.0, 10.0])))
    assert lo == "#084594"
    assert hi == "#EFF3FF"


def test_distiller_direction_and_fill():
    s = scale_fill_distiller(type="div", palette="RdBu", direction=1)
    assert s.aesthetics == ("fill",)
    s.train(np.array([0.0, 1.0]))
    lo, hi = hexes(s.map(np.array([0.0, 1.0])))
    assert (lo, hi) == (BREWER["RdBu"][7][0], BREWER["RdBu"][7][-1])


def test_distiller_qual_warns():
    with pytest.warns(UserWarning, match="qualitative"):
        scale_color_distiller(palette="Set2")


# --------------------------------------------------------------- fermenter
def test_fermenter_matches_ggplot():
    # ggplot2: scale_colour_fermenter(palette = "Greens") on y = 1:10 (breaks 2.5, 5, 7.5)
    s = scale_color_fermenter(palette="Greens", breaks=[2.5, 5, 7.5])
    assert isinstance(s, ScaleBinnedColour)
    s.train(np.arange(1, 11, dtype=float))
    a, b, c, d = "#238B45", "#74C476", "#BAE4B3", "#EDF8E9"
    expected = [a, a, b, b, b, c, c, d, d, d]
    assert hexes(s.map(np.arange(1, 11, dtype=float))) == expected
    assert list(s.bin_edges()) == [1.0, 2.5, 5.0, 7.5, 10.0]


def test_fermenter_missing_and_oob():
    s = scale_color_fermenter(palette="Greens", limits=[0, 10], breaks=[5])
    s.train(np.array([0.0, 10.0]))
    out = s.map(np.array([np.nan, 20.0, 3.0]))
    assert hexes(out[:2]) == ["#7F7F7F", "#7F7F7F"]
    assert hexes(out[2:]) == [BREWER["Greens"][3][1]]  # 2 bins -> first 2 of the 3-class scheme


def test_fermenter_too_many_bins():
    with pytest.warns(UserWarning, match="qualitative"):
        s = scale_color_fermenter(palette="Set1", breaks=list(range(1, 20)))
    s.train(np.array([0.0, 20.0]))
    with pytest.raises(ValueError, match="at most 9 colours"):
        s.map(np.array([1.0]))


# --------------------------------------------------------------- integration
def test_public_api_and_build():
    for name in ("brewer", "distiller", "fermenter"):
        for aes_ in ("color", "colour", "fill"):
            assert f"scale_{aes_}_{name}" in ggmanim.__all__
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "g": ["a", "b", "c"]})
    m = (
        ggplot(df, aes("x", "y", color="g", fill="y"))
        .geom_point()
        .scale_colour_brewer(type="qual", palette=2)
        .scale_fill_fermenter()
        .build()
    )
    assert m is not None
