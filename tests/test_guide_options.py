"""Guide options (ggm-4a9.4): guide_legend, merging, order, layout, placement."""

import numpy as np
import pandas as pd
import pytest
from manim import DL, UR, Square, VGroup

from ggmanim import aes, ggplot, guide_colorbar, guide_legend, guides, theme
from ggmanim.geoms import KEY_SIZE
from ggmanim.geoms.line import GeomLine
from ggmanim.geoms.point import GeomPoint
from ggmanim.guides import Colourbar, Guides, Legend, build_legends, place_legends
from ggmanim.guides_legend import GuideLegend, legend_frame
from ggmanim.scales.colour import scale_color_gradient
from ggmanim.scales.discrete import scale_color_discrete
from ggmanim.scales.shape import scale_shape
from ggmanim.theme import Theme

LEVELS = ["a", "b", "c", "d", "e"]


def trained(scale, values=LEVELS):
    scale.train(pd.Series(values))
    return scale


def legend(scales, layers=None, theme_=None, **kw):
    if layers is None:
        layers = [(GeomPoint(), {a: "g" for a in scales}, {})]
    return build_legends(scales, layers, theme_ or Theme(), **kw)


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "x": np.arange(10.0),
            "y": np.arange(10.0) ** 0.5,
            "g": list("abcde") * 2,
            "z": np.linspace(0, 1, 10),
        }
    )


def centres(mobs, axis):
    return sorted({round(float(m.get_center()[axis]), 6) for m in mobs})


# -- guide_legend object ---------------------------------------------------


def test_guide_legend_object_and_validation():
    g = guide_legend(title="T", ncol=2, override_aes={"colour": "red"}, order=1)
    assert isinstance(g, GuideLegend)
    assert g.override_aes == {"color": "red"}
    assert g == guide_legend(title="T", ncol=2, override_aes={"color": "red"}, order=1)
    assert hash(g) == hash(guide_legend(title="T", ncol=2, override_aes={"color": "red"}, order=1))
    assert "ncol=2" in repr(g)
    for bad in (dict(nrow=0), dict(ncol=1.5), dict(order=-1), dict(direction="diagonal")):
        with pytest.raises(ValueError):
            guide_legend(**bad)
    with pytest.raises(ValueError):
        guide_legend(nrow=1, ncol=2).grid(5, "vertical")


def test_guides_accepts_guide_objects_and_scale_guide():
    g = guides(color=guide_legend(ncol=2), fill=guide_colorbar())
    assert isinstance(g, Guides) and isinstance(g.guides["color"], GuideLegend)
    sc = trained(scale_color_discrete(guide=guide_legend(title="FromScale")))
    (leg,) = legend({"color": sc})
    assert leg.title.text == "FromScale"
    (leg,) = legend(
        {"color": trained(scale_color_discrete())}, guides={"color": guide_legend(title="G")}
    )
    assert leg.title.text == "G"


# -- merging -----------------------------------------------------------------


def test_colour_and_shape_on_one_column_merge_into_composite_keys():
    c, s = trained(scale_color_discrete()), trained(scale_shape())
    legs = legend({"color": c, "shape": s})
    assert len(legs) == 1
    (leg,) = legs
    assert set(leg.aesthetics) == {"color", "shape"}
    assert len(leg.keys) == 5
    # key glyphs carry both: the colour of the colour scale, the shape's marker type
    shapes = s.map(LEVELS)
    kinds = {type(k[0]).__name__ for k in leg.keys}
    assert len(kinds) > 1 or len(set(map(str, shapes))) == 1
    got = [k[0].get_fill_color().to_hex().upper() for k in leg.keys]
    assert got == [c_.to_hex().upper() for c_ in c.map(LEVELS)]


def test_no_merge_when_titles_differ():
    c = trained(scale_color_discrete())
    s = trained(scale_shape(name="Other"))
    legs = legend({"color": c, "shape": s})
    assert [l.aesthetics for l in legs] == [("color",), ("shape",)]
    # a guide title also breaks the merge
    legs = legend(
        {"color": trained(scale_color_discrete()), "shape": trained(scale_shape())},
        guides={"shape": guide_legend(title="S")},
    )
    assert len(legs) == 2


def test_merge_combines_override_aes():
    legs = legend(
        {"color": trained(scale_color_discrete()), "shape": trained(scale_shape())},
        guides={
            "color": guide_legend(override_aes={"size": 0.2}),
            "shape": guide_legend(override_aes={"size": 0.05, "alpha": 0.5}),
        },
    )
    (leg,) = legs
    assert leg.options.override_aes == {"size": 0.2, "alpha": 0.5}


# -- order ---------------------------------------------------------------------


def test_order_sorts_legends():
    c = trained(scale_color_discrete(name="C"))
    s = trained(scale_shape(name="S"))
    legs = legend({"color": c, "shape": s})
    assert [l.aesthetics for l in legs] == [("color",), ("shape",)]
    legs = legend({"color": c, "shape": s}, guides={"shape": guide_legend(order=1)})
    assert [l.aesthetics for l in legs] == [("shape",), ("color",)]
    legs = legend(
        {"color": c, "shape": s},
        guides={"shape": guide_legend(order=2), "color": guide_legend(order=1)},
    )
    assert [l.aesthetics for l in legs] == [("color",), ("shape",)]


def test_legend_and_colourbar_coexist_and_order(df):
    p = (
        ggplot(df, aes("x", "y", color="z", shape="g"))
        .geom_point()
        .add(guides(color=guide_colorbar(order=2), shape=guide_legend(order=1)))
    )
    m = p.build()
    assert [type(l) for l in m.legends] == [Legend, Colourbar]
    assert len(m.legend) == 2
    m2 = ggplot(df, aes("x", "y", color="z", shape="g")).geom_point().build()
    assert {type(l) for l in m2.legends} == {Legend, Colourbar}


# -- layout options ------------------------------------------------------------


def test_nrow_ncol_layout_counts():
    sc = trained(scale_color_discrete())
    (leg,) = legend({"color": sc}, guides={"color": guide_legend(ncol=2)})
    assert (leg.nrow, leg.ncol) == (3, 2)
    assert len(centres(leg.keys, 0)) == 2 and len(centres(leg.keys, 1)) == 3
    (leg,) = legend({"color": sc}, guides={"color": guide_legend(nrow=2)})
    assert (leg.nrow, leg.ncol) == (2, 3)
    assert len(centres(leg.keys, 0)) == 3 and len(centres(leg.keys, 1)) == 2
    (leg,) = legend({"color": sc})
    assert (leg.nrow, leg.ncol) == (5, 1)


def test_byrow_fill_order():
    sc = trained(scale_color_discrete())
    (col,) = legend({"color": sc}, guides={"color": guide_legend(ncol=2)})
    (row,) = legend({"color": sc}, guides={"color": guide_legend(ncol=2, byrow=True)})
    # column-major: the 2nd key sits under the first; byrow: right of it
    k0, k1 = col.keys[0].get_center(), col.keys[1].get_center()
    assert k1[0] == pytest.approx(k0[0]) and k1[1] < k0[1]
    k0, k1 = row.keys[0].get_center(), row.keys[1].get_center()
    assert k1[1] == pytest.approx(k0[1]) and k1[0] > k0[0]


def test_reverse_and_direction():
    sc = trained(scale_color_discrete())
    (leg,) = legend({"color": sc}, guides={"color": guide_legend(reverse=True)})
    assert [l.text for l in leg.labels] == LEVELS[::-1]
    assert leg.breaks == LEVELS[::-1]
    (leg,) = legend({"color": sc}, guides={"color": guide_legend(direction="horizontal")})
    place_legends(Square(4), [leg], Theme())  # right position, own direction wins
    assert leg.direction == "horizontal" and len(centres(leg.keys, 1)) == 1


def test_theme_legend_direction_independent_of_position():
    (leg,) = legend({"color": trained(scale_color_discrete())})
    place_legends(Square(4), [leg], Theme(legend_direction="horizontal"))
    assert len(centres(leg.keys, 1)) == 1 and len(centres(leg.keys, 0)) == 5
    place_legends(Square(4), [leg], Theme(legend_position="bottom", legend_direction="vertical"))
    assert len(centres(leg.keys, 0)) == 1


def test_override_aes_reaches_keys():
    sc = trained(scale_color_discrete())
    layers = [(GeomPoint(), {"color": "g"}, {"alpha": 0.2})]
    (leg,) = legend(
        {"color": sc}, layers, guides={"color": guide_legend(override_aes={"alpha": 1})}
    )
    assert leg.keys[0][0].get_fill_opacity() == pytest.approx(1.0)
    (leg,) = legend({"color": sc}, layers)
    assert leg.keys[0][0].get_fill_opacity() == pytest.approx(0.2)
    sizes = [0.05, 0.1, 0.15, 0.2, 0.25]
    (leg,) = legend(
        {"color": sc}, layers, guides={"color": guide_legend(override_aes={"size": sizes})}
    )
    assert [k[0].width for k in leg.keys] == pytest.approx([2 * s for s in sizes], abs=1e-6)


def test_show_legend_dict_per_aesthetic():
    c, s = trained(scale_color_discrete(name="C")), trained(scale_shape(name="S"))
    layers = [
        (GeomLine(), {"color": "g"}, {"show_legend": {"color": False}}),
        (GeomPoint(), {"color": "g", "shape": "g"}, {"show_legend": {"colour": False}}),
    ]
    legs = legend({"color": c, "shape": s}, layers)
    # nothing left for colour -> no colour legend; the point layer still shows shape
    assert [l.aesthetics for l in legs] == [("shape",)]
    layers[0] = (GeomLine(), {"color": "g"}, {"show_legend": {"color": True}})
    legs = legend({"color": c, "shape": s}, layers)
    assert [l.aesthetics for l in legs] == [("color",), ("shape",)]
    assert all(len(k) == 1 for k in legs[0].keys)


def test_linetype_keys_wider_and_key_size():
    from ggmanim.scales.shape import scale_linetype

    lt = trained(scale_linetype(), ["a", "b", "c"])
    (leg,) = legend({"linetype": lt}, [(GeomLine(), {"linetype": "g"}, {})])
    assert leg.keys[1].width == pytest.approx(2.5 * KEY_SIZE, rel=0.05)
    (leg,) = legend(
        {"linetype": lt}, [(GeomLine(), {"linetype": "g"}, {})], Theme(legend_key_width=0.5)
    )
    assert leg.keys[1].width == pytest.approx(0.5, rel=0.05)
    sc = trained(scale_color_discrete())
    (small,) = legend({"color": sc})
    (big,) = legend({"color": sc}, theme_=Theme(legend_key_size=1.0))
    gap = lambda l: l.keys[0].get_center()[1] - l.keys[1].get_center()[1]  # noqa: E731
    assert gap(big) > gap(small) + 0.5


def test_title_align_background_border_and_text_style():
    sc = trained(scale_color_discrete())
    th = Theme(
        legend_title_align=1.0,
        legend_background="#202020",
        legend_border=True,
        legend_border_color="#FF0000",
        legend_text_color="#00FF00",
    )
    (leg,) = legend({"color": sc}, theme_=th)
    rows = VGroup(*leg.rows)
    assert leg.title.get_right()[0] == pytest.approx(rows.get_right()[0], abs=1e-6)
    assert leg.box is not None and leg.submobjects[0] is leg.box
    assert leg.box.get_fill_color().to_hex().upper() == "#202020"
    assert leg.box.get_stroke_color().to_hex().upper() == "#FF0000"
    assert leg.labels[0].get_color().to_hex().upper() == "#00FF00"
    (plain,) = legend({"color": sc})
    assert plain.box is None and legend_frame(plain, Theme()) is None
    cb = build_legends(
        {"color": trained(scale_color_gradient(), [0.0, 1.0])},
        [(GeomPoint(), {"color": "z"}, {})],
        th,
    )[0]
    assert cb.box is not None and cb.submobjects[0] is cb.box


def test_math_labels_rendered():
    sc = trained(scale_color_discrete(labels=["$10^{3}$", "b", "c", "d", "e"]))
    (leg,) = legend({"color": sc})
    assert leg.labels[0].text == "10³"


# -- placement -----------------------------------------------------------------


def test_inside_placement_lands_within_panel(df):
    p = ggplot(df, aes("x", "y", color="g")).geom_point()
    for pos in [(1, 1), (0, 0), (0.5, 0.5), (0.95, 0.1)]:
        m = p.add(theme(legend_position=pos)).build()
        axes = m.panel.axes
        (ax0, ay0), (ax1, ay1) = axes.get_corner(DL)[:2], axes.get_corner(UR)[:2]
        (lx0, ly0), (lx1, ly1) = m.legend.get_corner(DL)[:2], m.legend.get_corner(UR)[:2]
        eps = 1e-6
        assert ax0 - eps <= lx0 and lx1 <= ax1 + eps and ay0 - eps <= ly0 and ly1 <= ay1 + eps
    # justification point lands on the position
    m = p.add(theme(legend_position=(0.5, 0.5), legend_justification="center")).build()
    assert np.allclose(m.legend.get_center()[:2], m.panel.axes.get_center()[:2], atol=1e-6)
    # inside legends take no room from the panel
    w_in = p.add(theme(legend_position=(1, 1))).build()
    w_out = p.build()
    assert w_in.panel.axes.width > w_out.panel.axes.width


def test_outside_justification_and_legend_box():
    sc = trained(scale_color_discrete(name="C"))
    sh = trained(scale_shape(name="S"))
    panel = Square(side_length=6)
    legs = legend({"color": sc, "shape": sh})
    g = place_legends(panel, legs, Theme(legend_justification="top"))
    assert g.get_top()[1] == pytest.approx(panel.get_top()[1])
    assert legs[0].get_top()[1] > legs[1].get_top()[1]  # stacked vertically
    g = place_legends(panel, legs, Theme(legend_box="horizontal", legend_spacing=0.7))
    assert legs[1].get_left()[0] == pytest.approx(legs[0].get_right()[0] + 0.7)
    g = place_legends(panel, legs, Theme(legend_position="bottom", legend_justification="left"))
    assert g.get_left()[0] == pytest.approx(panel.get_left()[0])
    g = place_legends(panel, legs, Theme(legend_box_spacing=1.0))
    assert g.get_left()[0] == pytest.approx(panel.get_right()[0] + 1.0)


def test_colourbar_direction_option():
    sc = trained(scale_color_gradient(), [0.0, 1.0])
    (cb,) = build_legends(
        {"color": sc},
        [(GeomPoint(), {"color": "z"}, {})],
        Theme(),
        guides={"color": guide_colorbar(direction="horizontal")},
    )
    place_legends(Square(4), [cb], Theme())
    assert cb.direction == "horizontal"
    with pytest.raises(ValueError):
        guide_colorbar(direction="up")


def test_horizontal_colourbar_labels_do_not_overlap():
    sc = trained(scale_color_gradient(), [0.0, 1.0])
    (cb,) = build_legends({"color": sc}, [(GeomPoint(), {"color": "z"}, {})], Theme())
    place_legends(Square(4), [cb], Theme(legend_position="bottom"))
    for a, b in zip(cb.labels, cb.labels[1:]):
        assert a.get_right()[0] < b.get_left()[0]


# -- theme validation ------------------------------------------------------------


def test_theme_legend_validation():
    assert Theme(legend_position=[0.2, 0.8]).legend_position == (0.2, 0.8)
    assert Theme(legend_position=(1, 1)).legend_inside
    assert Theme(legend_position=(1, 1)).legend_just() == (1.0, 1.0)
    assert Theme().legend_just() == (0.5, 0.5)
    assert Theme(legend_justification="top").legend_just() == (0.5, 1.0)
    assert Theme(legend_justification=[0, 1]).legend_just() == (0.0, 1.0)
    for bad in (
        dict(legend_position=(1, 2, 3)),
        dict(legend_position="middle"),
        dict(legend_justification="upper"),
        dict(legend_direction="diagonal"),
        dict(legend_box="grid"),
        dict(legend_key_size=-1),
        dict(legend_key_width=0),
        dict(legend_title_align=2),
        dict(legend_background="notacolour"),
    ):
        with pytest.raises(ValueError):
            Theme(**bad)
    assert theme(legend_position=(0.1, 0.9)).legend_position == (0.1, 0.9)
