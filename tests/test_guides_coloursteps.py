"""Structural tests for the colour-steps guide (ggmanim.guides_coloursteps)."""

import numpy as np
import pandas as pd
import pytest
from manim import ManimColor, Rectangle

import ggmanim
from ggmanim import aes, ggplot, guide_colorsteps, guide_coloursteps, guides
from ggmanim.geoms.point import GeomPoint
from ggmanim.guides import Colourbar, Coloursteps, GuideColoursteps, build_legends
from ggmanim.scales.brewer import scale_color_fermenter, scale_fill_fermenter
from ggmanim.scales.colour import scale_color_gradient
from ggmanim.theme import Theme


def hx(c):
    return ManimColor(c).to_hex().upper()


def build_one(scale, aes_name="color", **kw):
    (g,) = build_legends({aes_name: scale}, [(GeomPoint(), {aes_name: "z"}, {})], Theme(), **kw)
    return g


def fermenter(breaks=(2.5, 5, 7.5), limits=(1.0, 10.0), **kw):
    s = scale_color_fermenter(palette="Greens", breaks=list(breaks), **kw)
    s.train(pd.Series(limits))
    return s


def lengths(g):
    attr = "width" if g.direction == "horizontal" else "height"
    return [getattr(r, attr) for r in g.rects]


def centres(g, axis):
    return [r.get_center()[axis] for r in g.rects]


def test_default_guide_for_fermenter():
    s = scale_fill_fermenter()
    assert isinstance(s.guide, GuideColoursteps)
    assert s.guide.even_steps and not s.guide.show_limits


def test_steps_match_bins_and_palette():
    s = fermenter()
    g = build_one(s)
    assert isinstance(g, Coloursteps) and isinstance(g, Colourbar)
    assert len(g.rects) == len(s.bin_edges()) - 1 == 4
    assert all(isinstance(r, Rectangle) for r in g.rects)
    assert [hx(r.get_fill_color()) for r in g.rects] == [hx(c) for c in s.bin_colours()]
    # low to high, bottom to top
    ys = centres(g, 1)
    assert ys == sorted(ys)
    assert len(g.ticks) == 0  # no ticks by default


def test_labels_at_bin_boundaries():
    s = fermenter()
    g = build_one(s)
    assert g.breaks == [2.5, 5.0, 7.5]
    assert [lab.text for lab in g.labels] == ["2.5", "5.0", "7.5"]
    # each label is centred on the boundary between consecutive steps
    for lab, lower, upper in zip(g.labels, g.rects[:-1], g.rects[1:]):
        assert lower.get_top()[1] == pytest.approx(upper.get_bottom()[1])
        assert lab.get_center()[1] == pytest.approx(lower.get_top()[1])


def test_even_steps_geometry():
    # uneven bins: 1..2.5 (1.5), 2.5..5 (2.5), 5..7.5 (2.5), 7.5..10 (2.5)
    g = build_one(fermenter())
    ls = lengths(g)
    assert ls == pytest.approx([ls[0]] * 4)
    assert sum(ls) == pytest.approx(g.bar_length)


def test_uneven_steps_geometry():
    s = fermenter(breaks=(2, 3, 7), limits=(0.0, 10.0))
    g = build_one(s, guides={"color": guide_coloursteps(even_steps=False)})
    ls = np.array(lengths(g))
    widths = np.diff(s.bin_edges())
    assert ls / ls.sum() == pytest.approx(widths / widths.sum())
    # labels still on the boundaries
    for lab, r in zip(g.labels, g.rects[:-1]):
        assert lab.get_center()[1] == pytest.approx(r.get_top()[1])


def test_horizontal_and_reverse():
    s = fermenter()
    g = build_one(s, guides={"color": guide_coloursteps(direction="horizontal")})
    g.arrange_rows("horizontal")
    xs = centres(g, 0)
    assert xs == sorted(xs)
    for lab, lower, upper in zip(g.labels, g.rects[:-1], g.rects[1:]):
        assert lower.get_right()[0] == pytest.approx(upper.get_left()[0])
        assert lab.get_center()[0] == pytest.approx(lower.get_right()[0])
        assert lab.get_top()[1] < g.bar.get_bottom()[1]
    r = build_one(s, guides={"color": guide_coloursteps(reverse=True)})
    ys = centres(r, 1)
    assert ys == sorted(ys, reverse=True)
    assert r.labels[0].get_center()[1] > r.labels[-1].get_center()[1]


def test_show_limits_and_ticks():
    s = fermenter()
    g = build_one(s, guides={"color": guide_coloursteps(show_limits=True, ticks=True)})
    assert g.breaks == [1.0, 2.5, 5.0, 7.5, 10.0]
    assert [lab.text for lab in g.labels] == ["1.0", "2.5", "5.0", "7.5", "10.0"]
    assert g.labels[0].get_center()[1] == pytest.approx(g.bar.get_bottom()[1])
    assert g.labels[-1].get_center()[1] == pytest.approx(g.bar.get_top()[1])
    assert len(g.ticks) == 2 * 5


def test_show_limits_fixed_labels():
    s = fermenter(labels=["lo", "mid", "hi"])
    g = build_one(s, guides={"color": guide_coloursteps(show_limits=True)})
    assert [lab.text for lab in g.labels] == ["1", "lo", "mid", "hi", "10"]


def test_coloursteps_on_continuous_scale():
    s = scale_color_gradient(low="black", high="white")
    s.train(pd.Series([0.0, 10.0]))
    g = build_one(s, guides={"color": "colorsteps"})
    assert isinstance(g, Coloursteps)
    inner = [b for b in s.get_breaks() if 0 < b < 10]
    assert len(g.rects) == len(inner) + 1
    mids = (np.array(g.edges[:-1]) + np.array(g.edges[1:])) / 2
    assert [hx(r.get_fill_color()) for r in g.rects] == [hx(c) for c in s.map_transformed(mids)]


def test_colourbar_override_on_fermenter():
    g = build_one(fermenter(), guides={"color": "colourbar"})
    assert type(g) is Colourbar and len(g.rects) == 32


def test_guides_accepts_coloursteps_specs():
    for spec in ("coloursteps", "colorsteps", guide_coloursteps(), guide_colorsteps()):
        assert guides(fill=spec).guides["fill"] == spec
    assert guide_colorsteps is guide_coloursteps
    for name in ("guide_coloursteps", "guide_colorsteps"):
        assert name in ggmanim.__all__
    with pytest.raises(ValueError):
        guide_coloursteps(direction="diagonal")


def test_fermenter_plot_builds():
    df = pd.DataFrame({"x": np.arange(10.0), "y": np.arange(10.0), "z": np.arange(10.0)})
    m = ggplot(df, aes("x", "y", color="z")).geom_point().scale_color_fermenter().build()
    found = [s for s in m.get_family() if isinstance(s, Coloursteps)]
    assert len(found) == 1
