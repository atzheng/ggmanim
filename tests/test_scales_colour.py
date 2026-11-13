"""Continuous colour scales and gradient / viridis palettes (no mobjects)."""

import numpy as np
import pandas as pd
import pytest
from manim import ManimColor

from ggmanim.palettes import (
    PALETTES,
    VIRIDIS_TABLES,
    gradient_n_pal,
    oklab_to_rgb,
    resolve_palette,
    rgb_to_oklab,
    viridis_pal,
)
from ggmanim.registry import COMPONENTS
from ggmanim.scales.colour import (
    ScaleContinuousColour,
    scale_color_continuous,
    scale_color_gradient,
    scale_color_gradient2,
    scale_color_gradientn,
    scale_color_hue,
    scale_color_viridis_c,
    scale_color_viridis_d,
    scale_colour_gradient,
    scale_fill_gradient,
    scale_fill_viridis_c,
    scale_fill_viridis_d,
)
from ggmanim.scales.discrete import NA_COLOR, ScaleDiscrete, default_scale_for


def hx(c):
    return ManimColor(c).to_hex().upper()


def hexes(cols):
    return [hx(c) for c in cols]


def trained(scale, values=(0.0, 10.0)):
    scale.train(pd.Series(values))
    return scale


def close(a, b, tol=2):
    """Hex colours equal up to ``tol`` per 8-bit channel."""
    ra, rb = ManimColor(a).to_int_rgb(), ManimColor(b).to_int_rgb()
    return all(abs(int(x) - int(y)) <= tol for x, y in zip(ra, rb))


# -- Oklab -----------------------------------------------------------------


def test_oklab_round_trip_and_white():
    rgb = np.array([[1.0, 1.0, 1.0], [0.2, 0.5, 0.8], [0.0, 0.0, 0.0]])
    lab = rgb_to_oklab(rgb)
    assert lab[0, 0] == pytest.approx(1.0, abs=1e-4)
    assert lab[0, 1:] == pytest.approx([0, 0], abs=1e-4)
    assert oklab_to_rgb(lab) == pytest.approx(rgb, abs=1e-6)


# -- gradient palettes -------------------------------------------------------


def test_gradient_endpoints_and_midpoint():
    pal = gradient_n_pal(["#000000", "#FFFFFF"])
    lo, mid, hi = pal([0.0, 0.5, 1.0])
    assert hx(lo) == "#000000" and hx(hi) == "#FFFFFF"
    # perceptual midpoint: grey, lighter than linear-RGB 50% mixing would be dark
    r, g, b = ManimColor(mid).to_int_rgb()
    assert r == g == b and 90 < r < 140


def test_gradient_nan_and_clamping():
    pal = gradient_n_pal(["red", "blue"])
    out = pal([np.nan, -1.0, 2.0])
    assert out[0] is None
    assert hx(out[1]) == hx("red") and hx(out[2]) == hx("blue")


def test_gradient_values_positions():
    pal = gradient_n_pal(["#000000", "#FF0000", "#FFFFFF"], values=[0, 0.2, 1])
    assert hx(pal([0.2])[0]) == "#FF0000"
    with pytest.raises(ValueError):
        gradient_n_pal(["red", "blue"], values=[0, 0.5, 1])
    with pytest.raises(ValueError):
        gradient_n_pal(["red", "blue"], values=[1, 0])


def test_viridis_tables_and_palette():
    assert set(VIRIDIS_TABLES) == {"viridis", "magma", "inferno", "plasma", "cividis"}
    assert all(len(t) == 256 for t in VIRIDIS_TABLES.values())
    assert hexes(viridis_pal()(2)) == ["#440154", "#FDE725"]
    assert hexes(viridis_pal(direction=-1)(2)) == ["#FDE725", "#440154"]
    assert hexes(viridis_pal(option="A")(1)) == ["#000004"]  # magma
    firsts = {hx(viridis_pal(option=o)(3)[1]) for o in VIRIDIS_TABLES}
    assert len(firsts) == 5  # distinct palettes
    with pytest.raises(ValueError, match="Unknown viridis option"):
        viridis_pal(option="nope")
    assert len(resolve_palette("magma", 4)) == 4
    assert "viridis" in PALETTES


# -- ScaleContinuousColour -----------------------------------------------------


def test_default_gradient_endpoints():
    s = trained(scale_color_gradient())
    out = s.map(pd.Series([0.0, 10.0]))
    assert hexes(out) == ["#132B43", "#56B1F7"]
    assert isinstance(out, pd.Series)
    assert isinstance(s.map([5.0]), np.ndarray)


def test_na_and_oob():
    s = trained(scale_color_gradient(limits=(0, 5)))
    out = s.map([np.nan, 2.0, 7.0, None])
    assert out[0] == NA_COLOR and out[2] == NA_COLOR and out[3] == NA_COLOR
    assert out[1] != NA_COLOR
    sq = trained(scale_color_gradient(limits=(0, 5), oob="squish", low="black", high="white"))
    assert hx(sq.map([7.0])[0]) == "#FFFFFF"
    custom = trained(scale_color_gradient(limits=(0, 5), na_value="red"))
    assert hx(custom.map([np.nan])[0]) == hx("red")


def test_limits_override_training():
    s = trained(scale_color_gradient(low="black", high="white", limits=(0, 20)))
    assert s.get_limits() == (0.0, 20.0)
    r, g, b = ManimColor(s.map([10.0])[0]).to_int_rgb()
    assert r == g == b and 0 < r < 255


def test_zero_range_maps_to_middle():
    s = trained(scale_color_gradient(low="black", high="white"), values=[3.0, 3.0])
    mid = gradient_n_pal(["black", "white"])([0.5])[0]
    assert hx(s.map([3.0])[0]) == hx(mid)


def test_breaks_and_labels():
    s = trained(scale_color_gradient())
    assert list(s.get_breaks()) == [0, 2, 4, 6, 8, 10]
    assert s.get_labels() == ["0", "2", "4", "6", "8", "10"]
    s2 = trained(scale_color_gradient(breaks=[1, 5, 50], labels=["a", "b", "c"]))
    assert list(s2.get_breaks()) == [1, 5] and s2.get_labels() == ["a", "b"]


def test_log_transform():
    s = trained(scale_color_gradient(low="black", high="white", trans="log10"), [1, 100])
    mid = gradient_n_pal(["black", "white"])([0.5])[0]
    assert hx(s.map([10.0])[0]) == hx(mid)


def test_gradient2_midpoint():
    s = trained(scale_color_gradient2(low="blue", mid="white", high="red", midpoint=2), [0, 10])
    out = s.map([2.0, 10.0, 0.0])
    assert hx(out[0]) == "#FFFFFF"
    assert hx(out[1]) == hx("red")
    # 0 is 2 below the midpoint of an extent of 8: 3/8 of the way from blue to white
    assert hx(out[2]) == hx(gradient_n_pal(["blue", "white", "red"])([0.375])[0])


def test_gradientn():
    s = trained(scale_color_gradientn(colours=["red", "#00FF00", "blue"]))
    assert hexes(s.map([0.0, 5.0, 10.0])) == [hx("red"), hx("#00FF00"), hx("blue")]
    s2 = trained(scale_color_gradientn(colors=["red", "blue"], values=[0, 1]))
    assert hx(s2.map([0.0])[0]) == hx("red")
    with pytest.raises(TypeError):
        scale_color_gradientn()


def test_viridis_c_options_distinct():
    mids = set()
    for opt in ("viridis", "magma", "plasma", "inferno", "cividis"):
        s = trained(scale_color_viridis_c(option=opt))
        lo, hi = s.map([0.0, 10.0])
        assert hx(lo) == VIRIDIS_TABLES[opt][0] and hx(hi) == VIRIDIS_TABLES[opt][-1]
        mids.add(hx(s.map([5.0])[0]))
    assert len(mids) == 5
    # the interpolated midpoint is close to the true table midpoint
    s = trained(scale_color_viridis_c())
    assert close(hx(s.map([5.0])[0]), VIRIDIS_TABLES["viridis"][128], tol=4)
    rev = trained(scale_fill_viridis_c(direction=-1))
    assert hx(rev.map([0.0])[0]) == "#FDE725" and rev.aesthetics == ("fill",)


def test_continuous_type():
    assert hx(trained(scale_color_continuous()).map([0.0])[0]) == "#132B43"
    assert hx(trained(scale_color_continuous(type="viridis")).map([0.0])[0]) == "#440154"
    with pytest.raises(ValueError):
        scale_color_continuous(type="nope")


def test_clone_is_untrained_and_independent():
    s = trained(scale_color_gradient())
    c = s.clone()
    assert c.get_limits() is None and s.get_limits() == (0.0, 10.0)
    assert c.palette is s.palette


def test_aesthetics_and_aliases():
    assert scale_color_gradient().aesthetics == ("color",)
    assert scale_colour_gradient().aesthetics == ("color",)
    assert scale_fill_gradient().aesthetics == ("fill",)
    both = scale_color_gradient(aesthetics=("colour", "fill"))
    assert both.aesthetics == ("color", "fill")
    assert not both.is_discrete and both.guide == "colourbar"
    for name in (
        "scale_color_gradient",
        "scale_colour_gradient2",
        "scale_fill_gradientn",
        "scale_color_viridis_c",
        "scale_fill_viridis_d",
        "scale_colour_hue",
    ):
        assert name in COMPONENTS


def test_get_colours():
    s = trained(scale_color_gradient(low="black", high="white"))
    cols = s.get_colours(4)
    lum = [ManimColor(c).to_int_rgb()[0] for c in cols]
    assert len(cols) == 4 and lum == sorted(lum)
    assert scale_color_gradient().get_colours() == []


# -- discrete extras -----------------------------------------------------------


def test_viridis_d_and_hue():
    s = scale_color_viridis_d()
    s.train(["a", "b", "c"])
    assert isinstance(s, ScaleDiscrete)
    assert hexes(s.map(["a", "c"])) == ["#440154", "#FDE725"]
    f = scale_fill_viridis_d(option="magma", end=0.5)
    f.train(["x"])
    assert f.aesthetics == ("fill",) and hexes(f.map(["x"])) == ["#000004"]
    h = scale_color_hue(l=40)
    h.train(["a", "b"])
    dflt = default_scale_for("color", ["a", "b"])
    dflt.train(["a", "b"])
    assert hexes(h.map(["a"])) != hexes(dflt.map(["a"]))


def test_default_scale_selection_by_dtype():
    assert isinstance(default_scale_for("colour", pd.Series([1, 2, 3])), ScaleContinuousColour)
    assert isinstance(default_scale_for("fill", pd.Series([0.5])), ScaleContinuousColour)
    assert isinstance(default_scale_for("color", pd.Series([True, False])), ScaleDiscrete)
    assert isinstance(default_scale_for("color", pd.Series(["a"])), ScaleDiscrete)
    s = default_scale_for("color", [1.0])
    s.train([0.0, 1.0])
    assert hexes(s.map([0.0, 1.0])) == ["#132B43", "#56B1F7"]
