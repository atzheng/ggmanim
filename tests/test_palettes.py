import pytest
from manim import ManimColor

from ggmanim import palettes
from ggmanim.palettes import (
    BRIGHT,
    MUTED,
    PALETTES,
    VIBRANT,
    as_manim_color,
    hcl_to_hex,
    hsv_pal,
    hue_pal,
    resolve_palette,
)


@pytest.mark.parametrize("pal,n", [(BRIGHT, 8), (MUTED, 9), (VIBRANT, 7)])
def test_tol_palettes_are_plain_lists_of_manimcolor(pal, n):
    assert type(pal) is list
    assert len(pal) == n
    assert all(type(c) is ManimColor for c in pal)
    # behaves like a normal colour: str / hex work, len is the palette length
    assert pal[0].to_hex().startswith("#")


def test_hashable_manim_color_removed():
    assert not hasattr(palettes, "HashableManimColor")


def test_hue_pal_matches_ggplot2():
    # scales::hue_pal()(3) and (4)
    assert [c.to_hex() for c in hue_pal()(3)] == ["#F8766D", "#00BA38", "#619CFF"]
    assert [c.to_hex() for c in hue_pal()(4)] == ["#F8766D", "#7CAE00", "#00BFC4", "#C77CFF"]


def test_hue_pal_edge_cases():
    pal = hue_pal()
    assert pal(0) == []
    assert [c.to_hex() for c in pal(1)] == ["#F8766D"]
    cols = pal(12)
    assert len({c.to_hex() for c in cols}) == 12
    assert [c.to_hex() for c in hue_pal(direction=-1)(3)] == ["#619CFF", "#00BA38", "#F8766D"]


def test_hcl_to_hex_black_and_grey():
    assert hcl_to_hex(0, 0, 0) == "#000000"
    assert hcl_to_hex(0, 0, 100) == "#FFFFFF"


def test_hsv_pal():
    cols = hsv_pal()(5)
    assert len(cols) == 5 and all(isinstance(c, ManimColor) for c in cols)


def test_resolve_palette_list_name_callable():
    assert resolve_palette(BRIGHT, 3) == BRIGHT[:3]
    assert resolve_palette("muted", 2) == MUTED[:2]
    assert resolve_palette("Bright", 1) == BRIGHT[:1]
    assert len(resolve_palette(hue_pal(), 20)) == 20
    assert resolve_palette(BRIGHT, 0) == []
    assert set(PALETTES) >= {"hue", "bright", "muted", "vibrant"}


def test_resolve_palette_too_many_levels():
    with pytest.raises(ValueError, match="8 values but 9 levels"):
        resolve_palette(BRIGHT, 9)


def test_resolve_palette_cycle():
    out = resolve_palette(["a", "b"], 5, cycle=True)
    assert out == ["a", "b", "a", "b", "a"]


def test_resolve_palette_unknown_name():
    with pytest.raises(ValueError, match="Unknown palette"):
        resolve_palette("nope", 2)


def test_as_manim_color():
    c = ManimColor("#123456")
    assert as_manim_color(c) is c
    assert as_manim_color("#123456") == c


def test_legacy_package_still_imports():
    import ggmanim

    assert ggmanim.BRIGHT is BRIGHT
