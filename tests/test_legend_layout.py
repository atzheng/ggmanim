"""ggm-2oz: legend title position, legend_box_just, wrapping of long horizontal
legends, legend_text_hjust / legend_title_hjust, centred multi-line axis titles."""

import numpy as np
import pandas as pd
import pytest
from manim import Paragraph, VGroup

from ggmanim import aes, ggplot, guide_legend, guides, labs, theme
from ggmanim.geoms.point import GeomPoint
from ggmanim.guides import Legend, build_legends
from ggmanim.guides_legend import place_legend_title
from ggmanim.scales.colour import scale_color_gradient
from ggmanim.scales.discrete import scale_color_discrete
from ggmanim.scales.shape import scale_shape
from ggmanim.theme import Theme

LEVELS = ["a", "b", "c", "d", "e"]
EPS = 1e-6


def trained(scale, values=LEVELS):
    scale.train(pd.Series(values))
    return scale


def legend(scales, theme_=None):
    layers = [(GeomPoint(), {a: "g" for a in scales}, {})]
    return build_legends(scales, layers, theme_ or Theme())


def long_df(n=14):
    names = [f"category number {i}" for i in range(n)]
    return pd.DataFrame(
        {"x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float), "g": names}
    )


# -- theme fields -----------------------------------------------------------------


def test_theme_validation():
    assert Theme().legend_title_position == "top"
    for bad in ("middle", None, 1):
        with pytest.raises(ValueError, match="legend_title_position"):
            Theme(legend_title_position=bad)
    with pytest.raises(ValueError, match="legend_text_hjust"):
        Theme(legend_text_hjust=1.5)
    with pytest.raises(ValueError, match="legend_title_hjust"):
        Theme(legend_title_hjust=-0.1)
    with pytest.raises(ValueError, match="legend_box_just"):
        Theme(legend_box_just="sideways")
    assert Theme(legend_box_just=[1, 0]).legend_box_just == (1.0, 0.0)
    th = Theme()
    assert th.legend_box_justification("vertical") == 0.0
    assert th.legend_box_justification("horizontal") == 1.0
    th = Theme(legend_box_just="right")
    assert th.legend_box_justification("vertical") == 1.0
    assert th.legend_box_justification("horizontal") == 0.5
    # merged through theme()
    p = ggplot(long_df(3), aes("x", "y")).add(theme(legend_title_position="left"))
    assert p.theme.legend_title_position == "left"


def test_text_style_align():
    th = Theme()
    assert th.text_style("axis_title")["align"] == "center"
    assert th.text_style("legend_text")["align"] == "left"
    assert th.text_style("legend_title")["align"] == "left"
    assert "align" not in th.text_style("axis_text")
    assert Theme(legend_text_hjust=1).text_style("legend_text")["align"] == "right"
    assert Theme(legend_title_align=0.5).text_style("legend_title")["align"] == "center"
    th = Theme(legend_title_align=0.5, legend_title_hjust=1.0)
    assert th.text_style("legend_title")["align"] == "right"


# -- title position ---------------------------------------------------------------


def _body(guide):
    if isinstance(guide, Legend):
        return VGroup(*guide.rows)
    return VGroup(guide.bar, *guide.labels)


@pytest.mark.parametrize("direction", ["vertical", "horizontal"])
@pytest.mark.parametrize("kind", ["legend", "colourbar"])
def test_title_positions(kind, direction):
    if kind == "legend":
        scales = {"color": trained(scale_color_discrete())}
    else:
        scales = {"color": trained(scale_color_gradient(), [0.0, 1.0])}
    for pos in ("top", "bottom", "left", "right"):
        (g,) = legend(scales, Theme(legend_title_position=pos))
        g.arrange_rows(direction)
        body, t = _body(g), g.title
        if pos == "top":
            assert t.get_bottom()[1] > body.get_top()[1]
            assert t.get_left()[0] == pytest.approx(body.get_left()[0], abs=EPS)
        elif pos == "bottom":
            assert t.get_top()[1] < body.get_bottom()[1]
            assert t.get_left()[0] == pytest.approx(body.get_left()[0], abs=EPS)
        elif pos == "left":
            assert t.get_right()[0] < body.get_left()[0]
            assert t.get_center()[1] == pytest.approx(body.get_center()[1], abs=EPS)
        else:
            assert t.get_left()[0] > body.get_right()[0]
            assert t.get_center()[1] == pytest.approx(body.get_center()[1], abs=EPS)
        assert t in g.submobjects


def test_left_title_horizontal_legend_in_plot():
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0], "g": ["a", "b", "c"]})
    m = (
        ggplot(df, aes("x", "y", color="g"))
        .geom_point()
        .add(theme(legend_position="bottom", legend_title_position="left"))
        .build()
    )
    (leg,) = m.legends
    assert leg.direction == "horizontal" and leg.nrow == 1
    assert leg.title.get_right()[0] < VGroup(*leg.rows).get_left()[0]
    assert leg.title.get_center()[1] == pytest.approx(VGroup(*leg.rows).get_center()[1], abs=EPS)


def test_coloursteps_left_title():
    from ggmanim.scales.brewer import scale_color_fermenter

    sc = trained(scale_color_fermenter(), [0.0, 10.0])
    (cs,) = legend({"color": sc}, Theme(legend_title_position="left"))
    cs.arrange_rows("horizontal")
    assert cs.title.get_right()[0] < cs.bar.get_left()[0]


def test_title_hjust_overrides_align():
    sc = trained(scale_color_discrete())
    (leg,) = legend({"color": sc}, Theme(legend_title_align=0.0, legend_title_hjust=1.0))
    rows = VGroup(*leg.rows)
    assert leg.title.get_right()[0] == pytest.approx(rows.get_right()[0], abs=EPS)
    # placement helper on its own
    th = Theme(legend_title_hjust=0.5)
    body = VGroup(*leg.rows)
    t = place_legend_title(leg.title.copy(), body, th, 0.1)
    assert t.get_center()[0] == pytest.approx(body.get_center()[0], abs=EPS)


# -- legend_text_hjust ------------------------------------------------------------


def test_text_hjust_aligns_labels_in_column():
    sc = trained(scale_color_discrete(), ["a", "bbbbbbbb", "cc"])
    (left,) = legend({"color": sc})
    lefts = [lab.get_left()[0] for lab in left.labels]
    assert max(lefts) - min(lefts) < EPS
    (right,) = legend({"color": sc}, Theme(legend_text_hjust=1.0))
    rights = [lab.get_right()[0] for lab in right.labels]
    assert max(rights) - min(rights) < EPS
    (mid,) = legend({"color": sc}, Theme(legend_text_hjust=0.5))
    centres = [lab.get_center()[0] for lab in mid.labels]
    assert max(centres) - min(centres) < EPS
    # labels never run into the keys
    for key, lab in zip(right.keys, right.labels):
        assert lab.get_left()[0] > key.get_right()[0]


def test_text_hjust_colourbar_labels():
    sc = trained(scale_color_gradient(), [0.0, 1000.0])
    (cb,) = legend({"color": sc}, Theme(legend_text_hjust=1.0))
    rights = [lab.get_right()[0] for lab in cb.labels]
    assert max(rights) - min(rights) < EPS
    assert min(lab.get_left()[0] for lab in cb.labels) > cb.bar.get_right()[0]


def test_multiline_label_alignment():
    sc = trained(
        scale_color_discrete(labels=["short", "a much\nlonger line", "c"]), ["a", "b", "c"]
    )
    (leg,) = legend({"color": sc}, Theme(legend_text_hjust=1.0))
    para = leg.labels[1]
    assert isinstance(para, Paragraph)
    assert para[0].get_right()[0] == pytest.approx(para[1].get_right()[0], abs=0.02)


# -- legend_box_just --------------------------------------------------------------


def _two_legends(th):
    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 1.0, 2.0],
            "g": ["a", "b", "c"],
            "h": ["a much longer level name", "q", "r"],
        }
    )
    return ggplot(df, aes("x", "y", color="g", shape="h")).geom_point().add(th).build()


@pytest.mark.parametrize("just,edge", [(None, 0.0), ("right", 1.0), ("center", 0.5)])
def test_box_just_vertical(just, edge):
    m = _two_legends(theme(legend_box_just=just))
    a, b = m.legends
    assert abs(a.width - b.width) > 0.3
    pa = a.get_left()[0] + edge * a.width
    pb = b.get_left()[0] + edge * b.width
    assert pa == pytest.approx(pb, abs=EPS)


@pytest.mark.parametrize("just,edge", [(None, 1.0), ("bottom", 0.0), ((0.5, 0.5), 0.5)])
def test_box_just_horizontal(just, edge):
    m = _two_legends(theme(legend_box="horizontal", legend_box_just=just))
    a, b = m.legends
    assert a.get_right()[0] < b.get_left()[0]
    pa = a.get_bottom()[1] + edge * a.height
    pb = b.get_bottom()[1] + edge * b.height
    assert pa == pytest.approx(pb, abs=EPS)


# -- wrapping ---------------------------------------------------------------------


def test_long_bottom_legend_wraps():
    m = (
        ggplot(long_df(), aes("x", "y", color="g"))
        .geom_point()
        .add(theme(legend_position="bottom"))
        .build()
    )
    (leg,) = m.legends
    assert leg.nrow > 1
    assert leg.width <= m.panel.width + EPS
    # filled row by row: the first keys share the top row
    assert leg.keys[0].get_center()[1] == pytest.approx(leg.keys[1].get_center()[1], abs=EPS)
    # legend below the panel, not overlapping it
    assert leg.get_top()[1] < m.panel.get_bottom()[1]


def test_short_bottom_legend_does_not_wrap():
    m = (
        ggplot(long_df(3), aes("x", "y", color="g"))
        .geom_point()
        .add(theme(legend_position="bottom"))
        .build()
    )
    (leg,) = m.legends
    assert leg.nrow == 1


@pytest.mark.parametrize("opts", [{"nrow": 1}, {"ncol": 14}])
def test_fixed_grid_never_wraps(opts):
    m = (
        ggplot(long_df(), aes("x", "y", color="g"))
        .geom_point()
        .add(theme(legend_position="bottom"))
        .add(guides(color=guide_legend(**opts)))
        .build()
    )
    (leg,) = m.legends
    assert leg.nrow == 1 and leg.ncol == 14


def test_side_legends_do_not_wrap():
    m = (
        ggplot(long_df(), aes("x", "y", color="g"))
        .geom_point()
        .add(theme(legend_direction="horizontal"))
        .build()
    )
    (leg,) = m.legends
    assert leg.nrow == 1


def test_wide_legend_box_stacks_vertically():
    df = long_df(6)
    df["h"] = [f"another long level {i}" for i in range(6)]
    m = (
        ggplot(df, aes("x", "y", color="g", shape="h"))
        .geom_point()
        .add(theme(legend_position="bottom"))
        .add(scale_shape())
        .build()
    )
    a, b = m.legends
    assert a.get_bottom()[1] > b.get_top()[1]  # one above the other
    # explicit legend_box is respected
    m = (
        ggplot(df, aes("x", "y", color="g", shape="h"))
        .geom_point()
        .add(theme(legend_position="bottom", legend_box="horizontal"))
        .build()
    )
    a, b = m.legends
    assert a.get_right()[0] < b.get_left()[0]


def test_inside_horizontal_legend_wraps_to_area():
    m = (
        ggplot(long_df(), aes("x", "y", color="g"))
        .geom_point()
        .add(theme(legend_position=(0.5, 0.5), legend_direction="horizontal"))
        .build()
    )
    (leg,) = m.legends
    assert leg.nrow > 1 and leg.width <= m.panel.axes.width + EPS


# -- axis titles ------------------------------------------------------------------


def test_multiline_axis_title_centred():
    m = (
        ggplot(long_df(3), aes("x", "y"))
        .geom_point()
        .add(labs(x="a long first line\nshort", y="y title\nunits"))
        .build()
    )
    for title in (m.panel.x_title, m.panel.y_title):
        assert isinstance(title, Paragraph)
    xt = m.panel.x_title
    assert xt[0].get_center()[0] == pytest.approx(xt[1].get_center()[0], abs=0.02)
