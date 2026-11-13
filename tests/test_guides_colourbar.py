"""Structural tests for the colourbar guide (ggmanim.guides_colourbar)."""

import numpy as np
import pandas as pd
import pytest
from manim import Line, ManimColor, Rectangle

from ggmanim import aes, ggplot, guide_colorbar, guides, theme
from ggmanim.geoms.point import GeomPoint
from ggmanim.guides import Colourbar, Legend, build_legends, place_legends
from ggmanim.scales.colour import (
    scale_color_gradient,
    scale_color_viridis_c,
    scale_fill_gradient,
)
from ggmanim.theme import Theme


def hx(c):
    return ManimColor(c).to_hex().upper()


def trained(scale, values=(0.0, 10.0)):
    scale.train(pd.Series(values))
    return scale


@pytest.fixture
def df():
    return pd.DataFrame({"x": np.arange(6.0), "y": np.arange(6.0) ** 2, "z": np.linspace(0, 1, 6)})


def build_one(scale, aes_name="color", **kw):
    (cb,) = build_legends({aes_name: scale}, [(GeomPoint(), {aes_name: "z"}, {})], Theme(), **kw)
    return cb


def test_colourbar_glyphs_match_breaks():
    sc = trained(scale_color_gradient())
    cb = build_one(sc)
    assert isinstance(cb, Colourbar)
    breaks = list(sc.get_breaks())
    assert len(cb.labels) == len(breaks) == len(cb.breaks)
    assert [lab.text for lab in cb.labels] == sc.get_labels()
    assert len(cb.ticks) == 2 * len(breaks)
    assert all(isinstance(t, Line) for t in cb.ticks)
    assert cb.title is not None and cb.title.text == "z"
    assert cb.aesthetics == ("color",)


def test_bar_is_rectangles_low_to_high():
    sc = trained(scale_color_gradient(low="black", high="white"))
    cb = build_one(sc)
    assert len(cb.rects) == 32 and all(isinstance(r, Rectangle) for r in cb.rects)
    ys = [r.get_center()[1] for r in cb.rects]
    assert ys == sorted(ys)  # vertical, low values at the bottom
    lum = [ManimColor(r.get_fill_color()).to_int_rgb()[0] for r in cb.rects]
    assert lum == sorted(lum) and lum[0] < 30 and lum[-1] > 225
    assert all(hx(r.get_stroke_color()) == hx(r.get_fill_color()) for r in cb.rects)
    # bar is contiguous: total height == sum of cells
    assert cb.bar.height == pytest.approx(cb.bar_length)
    assert cb.bar.width == pytest.approx(cb.bar_thickness)


def test_labels_beside_bar_at_break_positions():
    sc = trained(scale_color_gradient())
    cb = build_one(sc)
    bar_right = cb.bar.get_right()[0]
    assert all(lab.get_left()[0] > bar_right for lab in cb.labels)
    ys = [lab.get_center()[1] for lab in cb.labels]
    assert ys == sorted(ys)
    # first/last break 0 and 10 sit at the ends of the bar
    assert ys[0] == pytest.approx(cb.bar.get_bottom()[1])
    assert ys[-1] == pytest.approx(cb.bar.get_top()[1])


def test_horizontal_layout():
    sc = trained(scale_color_gradient())
    cb = build_one(sc)
    cb.arrange_rows("horizontal")
    assert cb.direction == "horizontal"
    xs = [r.get_center()[0] for r in cb.rects]
    assert xs == sorted(xs)
    assert cb.bar.width == pytest.approx(cb.bar_length)
    bar_bottom = cb.bar.get_bottom()[1]
    assert all(lab.get_top()[1] < bar_bottom for lab in cb.labels)


def test_guide_colorbar_options():
    sc = trained(
        scale_color_gradient(
            guide=guide_colorbar(nbin=7, reverse=True, barheight=2.0, barwidth=0.5, title="T")
        )
    )
    cb = build_one(sc)
    assert len(cb.rects) == 7
    assert cb.bar.height == pytest.approx(2.0) and cb.bar.width == pytest.approx(0.5)
    assert cb.title.text == "T"
    ys = [lab.get_center()[1] for lab in cb.labels]
    assert ys == sorted(ys, reverse=True)  # reversed: high values at the bottom
    no_ticks = build_one(trained(scale_color_gradient(guide=guide_colorbar(ticks=False))))
    assert len(no_ticks.ticks) == 0
    with pytest.raises(ValueError):
        guide_colorbar(nbin=0)


def test_guides_component_overrides_scale_guide():
    sc = trained(scale_color_gradient())
    cb = build_one(sc, guides={"color": guide_colorbar(nbin=5)})
    assert isinstance(cb, Colourbar) and len(cb.rects) == 5
    assert build_legends({"color": sc}, [], Theme(), guides={"color": "none"}) == []


def test_legend_guide_gives_binned_keys():
    sc = trained(scale_color_gradient(guide="legend"))
    leg = build_one(sc)
    assert isinstance(leg, Legend)
    assert len(leg.keys) == len(sc.get_breaks())
    got = [hx(k[0].get_color()) for k in leg.keys]
    assert got == [hx(c) for c in sc.map(leg.breaks)]
    # log scale: legend breaks are in data space
    lg = trained(scale_color_gradient(guide="legend", trans="log10"), [1, 1000])
    leg = build_one(lg)
    assert leg.breaks == pytest.approx([1, 10, 100, 1000])


def test_colour_and_fill_same_column_merge():
    c = trained(scale_color_gradient())
    f = trained(scale_fill_gradient())
    out = build_legends(
        {"color": c, "fill": f}, [(GeomPoint(), {"color": "z", "fill": "z"}, {})], Theme()
    )
    assert len(out) == 1 and out[0].aesthetics == ("color", "fill")


def test_place_legends_mixes_legend_and_colourbar(df):
    p = ggplot(df.assign(g=list("aabbcc")), aes("x", "y", color="z", shape="g")).geom_point()
    m = p.build()
    kinds = sorted(type(leg).__name__ for leg in m.legends)
    assert kinds == ["Colourbar", "Legend"]
    assert m.legend.get_left()[0] > m.panel.get_right()[0]


def test_build_numeric_colour_shows_colourbar(df):
    m = ggplot(df, aes("x", "y", color="z")).geom_point().build()
    (cb,) = m.legends
    assert isinstance(cb, Colourbar)
    assert len(cb.labels) == len(m.scales["color"].get_breaks())
    pts = m.layers[0]
    cols = {hx(c) for c in m.layer_data[0]["color"]}
    assert len(cols) == 6 and "#132B43" in cols and "#56B1F7" in cols
    assert len(pts) == 6


def test_build_colour_and_fill_numeric_one_colourbar(df):
    m = ggplot(df, aes("x", "y", color="z", fill="z")).geom_point().build()
    assert len(m.legends) == 1 and isinstance(m.legends[0], Colourbar)
    assert m.legends[0].aesthetics == ("color", "fill")


def test_build_top_legend_is_horizontal(df):
    p = ggplot(df, aes("x", "y", color="z")).geom_point().add(theme(legend_position="bottom"))
    m = p.build()
    (cb,) = m.legends
    assert cb.direction == "horizontal"
    assert cb.get_top()[1] < m.panel.get_bottom()[1]


def test_build_viridis_and_guides_none(df):
    m = ggplot(df, aes("x", "y", color="z")).geom_point().add(scale_color_viridis_c()).build()
    assert hx(m.layer_data[0]["color"].iloc[0]) == "#440154"
    m2 = ggplot(df, aes("x", "y", color="z")).geom_point().add(guides(color="none")).build()
    assert m2.legends == []


def test_place_legends_direct():
    sc = trained(scale_color_gradient())
    cb = build_one(sc)
    from manim import Square

    panel = Square(4)
    g = place_legends(panel, [cb], Theme(legend_position="top"))
    assert cb.direction == "horizontal" and g.get_bottom()[1] > panel.get_top()[1]
