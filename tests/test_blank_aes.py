"""Blank fill/colour constants (ggm-816): ``fill=None`` / ``"NA"`` / ``"none"``
override a mapped aesthetic and the geom default, as ggplot2's ``fill = NA``."""

import numpy as np
import pandas as pd
import pytest

from ggmanim import (
    aes,
    geom_abline,
    geom_area,
    geom_boxplot,
    geom_col,
    geom_crossbar,
    geom_density,
    geom_hline,
    geom_label,
    geom_line,
    geom_path,
    geom_point,
    geom_polygon,
    geom_raster,
    geom_rect,
    geom_ribbon,
    geom_rug,
    geom_segment,
    geom_smooth,
    geom_text,
    geom_tile,
    geom_violin,
    geom_vline,
    ggplot,
)
from ggmanim.geoms import GeomDensity, GeomPoint, GeomPolygon
from ggmanim.geoms._markers import make_marker
from ggmanim.geoms.base import NA_COLOR, is_blank, resolve_color

RNG = np.random.default_rng(1)
DATA = pd.DataFrame(
    {
        "x": np.tile(np.arange(1.0, 6.0), 2),
        "y": RNG.normal(size=10) + 3,
        "g": np.repeat(["a", "b"], 5),
    }
)
DATA["lo"], DATA["hi"] = DATA["y"] - 1, DATA["y"] + 1
DATA["xend"], DATA["yend"] = DATA["x"] + 0.5, DATA["y"] + 0.5
DATA["label"] = list("abcdefghij")


def _leaves(mob):
    return [m for m in mob.get_family() if len(m.points)]


def has_fill(mob) -> bool:
    return any(float(m.get_fill_opacity()) > 0 for m in _leaves(mob))


def has_stroke(mob) -> bool:
    return any(
        float(m.get_stroke_width()) > 0 and float(m.get_stroke_opacity()) > 0 for m in _leaves(mob)
    )


def layer(plot, i=0):
    return plot.build().layers[i]


# -- the constants ------------------------------------------------------------------


@pytest.mark.parametrize("v", ["NA", "na", " NA ", "none", "None", "transparent"])
def test_is_blank_strings(v):
    assert is_blank(v)
    assert resolve_color(v, "#ff0000") is None  # blank beats the fallback


@pytest.mark.parametrize("v", [None, "red", "#000000", float("nan"), 0])
def test_not_blank(v):
    assert not is_blank(v)


@pytest.mark.parametrize("v", [None, "NA", "none"])
def test_aes_params_keep_blank_colours(v):
    params = GeomPolygon().aes_params({"fill": v, "colour": v, "alpha": None, "size": 0.1})
    assert params["fill"] == NA_COLOR
    assert params["color"] == NA_COLOR  # colour alias standardised
    assert "alpha" not in params  # None still means "not set" for other aesthetics


# -- the bead's case ----------------------------------------------------------------


@pytest.mark.parametrize("blank", [None, "NA", "none"])
def test_density_fill_none_overrides_mapping(blank):
    filled = layer(ggplot(DATA, aes("y", fill="g")).add(geom_density(alpha=0.4)))
    assert has_fill(filled)
    lay = layer(ggplot(DATA, aes("y", fill="g")).add(geom_density(fill=blank)))
    assert len(lay.marks) == 2  # still grouped by the mapped fill
    assert not has_fill(lay)
    assert has_stroke(lay)  # the outline stays


def test_density_colour_blank_drops_outline():
    lay = layer(ggplot(DATA, aes("y", fill="g")).add(geom_density(color="NA")))
    assert has_fill(lay)
    assert not has_stroke(lay)


# -- fill = NA on every filled geom -------------------------------------------------

FILLED = {
    "polygon": lambda f: geom_polygon(aes(group="g"), fill=f),
    "rect": lambda f: geom_rect(aes(xmin="x", xmax="xend", ymin="lo", ymax="hi", fill="g"), fill=f),
    "tile": lambda f: geom_tile(aes(fill="g"), fill=f),
    "raster": lambda f: geom_raster(aes(fill="y"), fill=f),
    "area": lambda f: geom_area(aes(fill="g"), fill=f),
    "ribbon": lambda f: geom_ribbon(aes(ymin="lo", ymax="hi", fill="g"), fill=f),
    "col": lambda f: geom_col(aes(fill="g"), fill=f, fast=True),
    "col_slow": lambda f: geom_col(aes(fill="g"), fill=f, fast=False),
    "boxplot": lambda f: geom_boxplot(aes("g", "y", fill="g"), fill=f, outliers=False),
    "notched": lambda f: geom_boxplot(aes("g", "y", fill="g"), fill=f, notch=True, outliers=False),
    "violin": lambda f: geom_violin(aes("g", "y", fill="g"), fill=f),
    "crossbar": lambda f: geom_crossbar(aes(ymin="lo", ymax="hi", fill="g"), fill=f),
    "point21": lambda f: geom_point(aes(fill="g"), shape=21, fill=f),
    "smooth": lambda f: geom_smooth(aes(fill="g"), method="lm", fill=f),
}


@pytest.mark.parametrize("name", sorted(FILLED))
def test_fill_blank_on_filled_geoms(name):
    make = FILLED[name]
    base = ggplot(DATA, aes("x", "y"))
    assert has_fill(layer(base.add(make("#3366cc"))))
    assert not has_fill(layer(base.add(make(None)))), name
    assert not has_fill(layer(base.add(make("none")))), name


# -- colour = NA on every stroked geom ----------------------------------------------

STROKED = {
    "line": lambda c: geom_line(aes(color="g"), color=c),
    "path": lambda c: geom_path(aes(color="g"), color=c, linetype="dashed"),
    "segment": lambda c: geom_segment(aes(xend="xend", yend="yend", color="g"), color=c),
    "segment_fast": lambda c: geom_segment(aes(xend="xend", yend="yend"), color=c, fast=True),
    "abline": lambda c: geom_abline(intercept=1, slope=0.5, color=c),
    "hline": lambda c: geom_hline(yintercept=3, color=c),
    "vline": lambda c: geom_vline(xintercept=3, color=c),
    "rug": lambda c: geom_rug(aes(color="g"), color=c),
    "point": lambda c: geom_point(aes(color="g"), color=c),
    "point_open": lambda c: geom_point(shape=1, color=c),
    "point_fast": lambda c: geom_point(aes(color="g"), color=c, fast=True),
    "polygon": lambda c: geom_polygon(aes(group="g"), color=c, fill="NA"),
    "area": lambda c: geom_area(aes(group="g"), color=c, fill="NA"),
    "boxplot": lambda c: geom_boxplot(aes("g", "y"), color=c, outliers=False),
    "violin": lambda c: geom_violin(aes("g", "y"), color=c),
    "crossbar": lambda c: geom_crossbar(aes(ymin="lo", ymax="hi"), color=c),
    "smooth": lambda c: geom_smooth(method="lm", se=False, color=c),
    "text": lambda c: geom_text(aes(label="label"), color=c),
}


@pytest.mark.parametrize("name", sorted(STROKED))
def test_colour_blank_on_stroked_geoms(name):
    make = STROKED[name]
    base = ggplot(DATA, aes("x", "y")).add(geom_point())  # trains the scales for abline

    def visible(lay):
        return has_stroke(lay) or has_fill(lay)

    assert visible(layer(base.add(make("#cc3333")), 1))
    assert not visible(layer(base.add(make(None)), 1)), name
    assert not visible(layer(base.add(make("NA")), 1)), name


def test_point_solid_with_blank_colour_uses_fill():
    m = make_marker(19, np.zeros(3), 0.1, color=None, fill=resolve_color("red"))
    assert has_fill(m) and not has_stroke(m)
    assert not has_fill(make_marker(19, np.zeros(3), 0.1, color=None))


@pytest.mark.parametrize("blank", [None, "NA"])
def test_label_blank_fill_drops_box(blank):
    lay = layer(ggplot(DATA, aes("x", "y")).add(geom_label(aes(label="label"), fill=blank)))
    for mark in lay.marks:
        assert not has_fill(mark[0])  # no background box
        assert has_fill(mark[1])  # the text is still drawn


def test_label_blank_colour_keeps_box():
    lay = layer(ggplot(DATA, aes("x", "y")).add(geom_label(aes(label="label"), color="NA")))
    assert has_fill(lay)  # the background box
    assert not has_stroke(lay)  # no border
    for mark in lay.marks:
        assert float(mark[1].get_fill_opacity()) == 0  # no text


# -- legend keys and defaults -------------------------------------------------------


def test_keys_honour_blank():
    assert not has_fill(GeomDensity().draw_key({"fill": "NA"}))
    assert not has_fill(GeomPolygon().draw_key({"fill": "NA"}))
    key = GeomPoint().draw_key({"color": "NA"})
    assert not has_fill(key) and not has_stroke(key)


def test_legend_reflects_blank_constant():
    shown = ggplot(DATA, aes("y", fill="g")).add(geom_density()).build()
    assert all(has_fill(key) for key in shown.legends[0].keys)
    p = ggplot(DATA, aes("y", fill="g")).add(geom_density(fill=None)).build()
    assert len(p.legends) == 1
    assert len(p.legends[0].keys) == 2
    for key in p.legends[0].keys:
        assert not has_fill(key)


def test_polygon_default_fill_blanked():
    # geom_polygon fills grey by default; fill=None (as in the radar docs) removes it
    lay = layer(ggplot(DATA, aes("x", "y")).add(geom_polygon(fill=None, color="red")))
    assert not has_fill(lay)
    assert has_stroke(lay)
