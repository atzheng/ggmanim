"""Structural tests for legends (ggmanim.guides)."""

import pandas as pd
import pytest
from manim import Square, Text, VGroup

from ggmanim.aes import aes
from ggmanim.geoms.col import GeomCol
from ggmanim.geoms.line import GeomLine
from ggmanim.geoms.point import GeomPoint
from ggmanim.guides import Guides, Legend, build_legends, guides, place_legends
from ggmanim.plot import ggplot
from ggmanim.registry import COMPONENTS
from ggmanim.scales.discrete import ScaleDiscrete, scale_color_discrete, scale_fill_discrete
from ggmanim.theme import Theme

LEVELS = ["a", "b", "c"]


def trained(scale, values=LEVELS):
    scale.train(pd.Series(values))
    return scale


def colour_scale(**kw):
    return trained(scale_color_discrete(**kw))


def fill_scale(**kw):
    return trained(scale_fill_discrete(**kw))


def hexes(colors):
    return [c.to_hex().upper() for c in colors]


# -- build_legends ----------------------------------------------------------


def test_row_count_equals_levels_and_attributes():
    sc = colour_scale()
    (leg,) = build_legends({"color": sc}, [(GeomPoint(), {"color": "grp"}, {})], Theme())
    assert isinstance(leg, Legend)
    assert len(leg.rows) == len(leg.keys) == len(leg.labels) == 3
    assert leg.aesthetics == ("color",)
    assert [l.text for l in leg.labels] == LEVELS
    assert leg.title is not None and leg.title.text == "grp"
    # rows are stacked top to bottom
    ys = [r.get_center()[1] for r in leg.rows]
    assert ys == sorted(ys, reverse=True)


def test_glyph_colour_matches_scale_map():
    sc = colour_scale()
    (leg,) = build_legends({"color": sc}, [(GeomPoint(), {"color": "grp"}, {})], Theme())
    expected = hexes(sc.map(LEVELS))
    got = [k[0].get_color().to_hex().upper() for k in leg.keys]
    assert got == expected


def test_line_glyph_colour():
    sc = colour_scale()
    (leg,) = build_legends({"color": sc}, [(GeomLine(), {"color": "grp"}, {})], Theme())
    got = [k[0].get_stroke_color().to_hex().upper() for k in leg.keys]
    assert got == hexes(sc.map(LEVELS))


def test_several_layers_overlay_glyphs():
    sc = colour_scale()
    layers = [
        (GeomLine(), {"color": "grp"}, {}),
        (GeomPoint(), {"color": "grp"}, {}),
        (GeomPoint(), {"x": "x"}, {}),
    ]  # last layer does not map colour
    (leg,) = build_legends({"color": sc}, layers, Theme())
    assert all(len(k) == 2 for k in leg.keys)


def test_show_legend_false_skips_layer():
    sc = colour_scale()
    layers = [
        (GeomLine(), {"color": "grp"}, {"show_legend": False}),
        (GeomPoint(), {"color": "grp"}, {}),
    ]
    (leg,) = build_legends({"color": sc}, layers, Theme())
    assert all(len(k) == 1 for k in leg.keys)


def test_constant_params_reach_glyph():
    sc = colour_scale()
    (leg,) = build_legends({"color": sc}, [(GeomPoint(), {"color": "grp"}, {"size": 0.1})], Theme())
    assert leg.keys[0][0].width == pytest.approx(0.2, abs=1e-6)


def test_colour_and_fill_on_same_column_merge():
    c, f = colour_scale(), fill_scale(palette=["#FF0000", "#00FF00", "#0000FF"])
    layers = [(GeomCol(), {"color": "grp", "fill": "grp"}, {})]
    legs = build_legends({"color": c, "fill": f}, layers, Theme())
    assert len(legs) == 1
    (leg,) = legs
    assert set(leg.aesthetics) == {"color", "fill"}
    assert len(leg.rows) == 3
    fills = [k[0].get_fill_color().to_hex().upper() for k in leg.keys]
    assert fills == ["#FF0000", "#00FF00", "#0000FF"]


def test_different_columns_give_two_legends():
    c = colour_scale()
    f = trained(scale_fill_discrete(), ["x", "y"])
    layers = [(GeomCol(), {"color": "grp", "fill": "kind"}, {})]
    legs = build_legends({"color": c, "fill": f}, layers, Theme())
    assert [l.aesthetics for l in legs] == [("color",), ("fill",)]
    assert [len(l.rows) for l in legs] == [3, 2]


def test_shared_scale_object_under_two_keys_is_one_legend():
    sc = trained(ScaleDiscrete(("color", "fill")))
    legs = build_legends(
        {"color": sc, "fill": sc}, [(GeomPoint(), {"color": "g", "fill": "g"}, {})], Theme()
    )
    assert len(legs) == 1 and legs[0].aesthetics == ("color", "fill")


def test_guides_none_and_scale_guide_none_remove_legend():
    sc = colour_scale()
    layers = [(GeomPoint(), {"color": "grp"}, {})]
    assert build_legends({"color": sc}, layers, Theme(), guides={"color": "none"}) == []
    assert build_legends({"colour": sc}, layers, Theme(), guides={"colour": "none"}) == []
    assert build_legends({"color": colour_scale(guide="none")}, layers, Theme()) == []


def test_non_discrete_scales_skipped():
    class Cont:
        is_discrete = False
        guide = "legend"

    assert build_legends({"color": Cont()}, [], Theme()) == []


def test_labels_use_text_engine_by_default():
    sc = colour_scale()
    (leg,) = build_legends({"color": sc}, [(GeomPoint(), {"color": "grp"}, {})], Theme())
    assert all(isinstance(l, Text) for l in leg.labels)
    assert isinstance(leg.title, Text)


def test_title_priority():
    layers = [(GeomPoint(), {"color": "grp"}, {})]
    (leg,) = build_legends(
        {"color": colour_scale(name="Scale")}, layers, Theme(), labels={"color": "Lab"}
    )
    assert leg.title.text == "Scale"
    (leg,) = build_legends({"color": colour_scale()}, layers, Theme(), labels={"colour": "Lab"})
    assert leg.title.text == "Lab"
    (leg,) = build_legends({"color": colour_scale()}, layers, Theme(), labels={"color": None})
    assert leg.title is None
    mob = Square()
    (leg,) = build_legends({"color": colour_scale()}, layers, Theme(), labels={"color": mob})
    assert isinstance(leg.title, Square) and leg.title is not mob


def test_scale_labels_and_breaks_used():
    sc = colour_scale(breaks=["c", "a"], labels={"a": "Alpha"})
    (leg,) = build_legends({"color": sc}, [(GeomPoint(), {"color": "grp"}, {})], Theme())
    assert [l.text for l in leg.labels] == ["c", "Alpha"]
    assert hexes([k[0].get_color() for k in leg.keys]) == hexes(sc.map(["c", "a"]))


def test_layer_objects_accepted():
    sc = colour_scale()
    layer = COMPONENTS["geom_point"](aes(color="grp"))
    (leg,) = build_legends({"color": sc}, [layer], Theme())
    assert len(leg.keys[0]) == 1


def test_fallback_square_when_no_layer():
    sc = colour_scale()
    (leg,) = build_legends({"color": sc}, [], Theme())
    assert isinstance(leg.keys[0][0], Square)


# -- placement ---------------------------------------------------------------


def _panel():
    return Square(side_length=4)


@pytest.mark.parametrize("pos", ["right", "left", "top", "bottom"])
def test_place_legends_positions(pos):
    panel = _panel()
    before = panel.get_center().copy()
    legs = build_legends(
        {"color": colour_scale(), "fill": trained(scale_fill_discrete(), ["x"])},
        [(GeomCol(), {"color": "grp", "fill": "kind"}, {})],
        Theme(),
    )
    placed = place_legends(panel, legs, Theme(legend_position=pos))
    assert isinstance(placed, VGroup) and len(placed) == 2
    assert (panel.get_center() == before).all()
    if pos == "right":
        assert placed.get_left()[0] > panel.get_right()[0]
    elif pos == "left":
        assert placed.get_right()[0] < panel.get_left()[0]
    elif pos == "top":
        assert placed.get_bottom()[1] > panel.get_top()[1]
    else:
        assert placed.get_top()[1] < panel.get_bottom()[1]


def test_bottom_legend_rows_horizontal():
    (leg,) = build_legends({"color": colour_scale()}, [(GeomPoint(), {"color": "g"}, {})], Theme())
    place_legends(_panel(), [leg], Theme(legend_position="bottom"))
    xs = [r.get_center()[0] for r in leg.rows]
    assert xs == sorted(xs)
    ys = [r.get_center()[1] for r in leg.rows]
    assert max(ys) - min(ys) < 1e-6 + 0.2


def test_place_legends_none():
    legs = build_legends({"color": colour_scale()}, [(GeomPoint(), {"color": "g"}, {})], Theme())
    placed = place_legends(_panel(), legs, Theme(legend_position="none"))
    assert isinstance(placed, VGroup) and len(placed) == 0
    assert len(place_legends(_panel(), [], Theme())) == 0


# -- guides component --------------------------------------------------------


def test_guides_component_on_plot():
    assert COMPONENTS["guides"] is guides
    p = ggplot(pd.DataFrame({"g": LEVELS}), aes(color="g"))
    assert dict(p.guides) == {}
    q = p.add(guides(colour="none"))
    assert dict(q.guides) == {"color": "none"} and dict(p.guides) == {}
    r = q.guides(fill="none")  # chaining sugar; p.guides is also the mapping
    assert dict(r.guides) == {"color": "none", "fill": "none"}
    s = r.add(guides(color="legend"))
    assert s.guides["color"] == "legend"
    with pytest.raises(TypeError):
        s.guides["x"] = "none"
    with pytest.raises(TypeError):
        s.guide_specs["x"] = "none"
    assert (
        build_legends(
            {"color": colour_scale()}, [(GeomPoint(), {"color": "g"}, {})], Theme(), guides=q.guides
        )
        == []
    )


def test_guides_validation():
    assert isinstance(guides(color=False), Guides)
    assert guides(color=False).guides == {"color": "none"}
    with pytest.raises(ValueError):
        guides(color="colourbar")
