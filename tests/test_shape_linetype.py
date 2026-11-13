"""Mapped / constant ``shape`` and ``linetype``: markers, dashes, scales, legends."""

import math

import numpy as np
import pandas as pd
import pytest
from manim import ORIGIN, RED, Cross, Dot, Line, RegularPolygon, Square, Star, Triangle, VGroup

import ggmanim
from ggmanim import (
    LINETYPES,
    SHAPES,
    aes,
    ggplot,
    scale_linetype,
    scale_linetype_manual,
    scale_shape,
    scale_shape_manual,
)
from ggmanim.geoms import KEY_SIZE, GeomLine, GeomPoint, GeomSmooth, GeomVline
from ggmanim.geoms._linetype import DASH_UNIT, apply_linetype, dash_polyline
from ggmanim.geoms._markers import make_marker
from ggmanim.geoms.base import path_mobject
from ggmanim.scales.shape import SHAPES_OPEN, parse_linetype, parse_shape


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
            "g": ["a", "a", "b", "b", "c", "c"],
        }
    )


def _area(m) -> float:
    if isinstance(m, Dot):
        return math.pi * m.radius**2
    v = m.get_vertices()
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# -- markers ------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, cls",
    [
        ("circle", Dot),
        ("triangle", Triangle),
        ("square", Square),
        ("diamond", RegularPolygon),
        ("star", Star),
        ("cross", Cross),
        ("plus", Cross),
    ],
)
def test_marker_classes(shape, cls):
    m = make_marker(shape, np.array([1.0, 2.0, 0.0]), 0.1, RED)
    assert isinstance(m, cls)
    centre = m.get_vertices().mean(axis=0) if not isinstance(m, (Dot, Cross)) else m.get_center()
    np.testing.assert_allclose(centre[:2], [1.0, 2.0], atol=1e-6)


@pytest.mark.parametrize("shape", ["triangle", "square", "diamond", "star"])
def test_markers_equal_area(shape):
    r = 0.1
    m = make_marker(shape, ORIGIN, r, RED)
    assert _area(m) == pytest.approx(math.pi * r**2, rel=1e-6)


def test_open_and_filled_styles():
    solid = make_marker("square", ORIGIN, 0.1, RED)
    assert solid.get_fill_opacity() == 1 and solid.get_stroke_width() == 0
    hollow = make_marker("square_open", ORIGIN, 0.1, RED, fill="#00FF00")
    assert hollow.get_fill_opacity() == 0 and hollow.get_stroke_width() > 0
    assert hollow.get_stroke_color().to_hex().upper() == RED.to_hex().upper()
    filled = make_marker("circle_filled", ORIGIN, 0.1, RED, fill="#00FF00", alpha=0.5)
    assert filled.get_fill_color().to_hex().upper() == "#00FF00"
    assert filled.get_stroke_color().to_hex().upper() == RED.to_hex().upper()
    assert filled.get_fill_opacity() == pytest.approx(0.5)


def test_parse_shape_aliases_and_pch():
    assert parse_shape(None) == ("circle", "solid")
    assert parse_shape(float("nan")) == ("circle", "solid")
    assert parse_shape(1) == ("circle", "open")
    assert parse_shape(17) == ("triangle", "solid")
    assert parse_shape("Diamond_Open") == ("diamond", "open")
    assert parse_shape("x") == ("cross", "stroke")


def test_unknown_shape_error_lists_shapes():
    with pytest.raises(ValueError, match="Valid shapes") as e:
        parse_shape("hexagon")
    for s in SHAPES:
        assert s in str(e.value)


# -- geom_point ---------------------------------------------------------------


def test_point_three_shapes_three_types(df):
    m = ggplot(df, aes("x", "y", shape="g")).geom_point().build()
    types = [type(d) for d in m.layers[0]]
    assert types == [Dot, Dot, Triangle, Triangle, Square, Square]
    assert list(m.scales["shape"].get_palette_values()) == SHAPES[:3]


def test_point_constant_shape(df):
    m = ggplot(df, aes("x", "y")).geom_point(shape="star_open").build()
    assert all(isinstance(d, Star) for d in m.layers[0])
    assert all(d.get_fill_opacity() == 0 for d in m.layers[0])


def test_point_constant_shape_overrides_mapping(df):
    m = ggplot(df, aes("x", "y", shape="g")).geom_point(shape="diamond").build()
    assert all(isinstance(d, RegularPolygon) for d in m.layers[0])


def test_scale_shape_solid_false(df):
    m = ggplot(df, aes("x", "y", shape="g")).geom_point().scale_shape(solid=False).build()
    assert list(m.scales["shape"].get_palette_values()) == SHAPES_OPEN[:3]
    assert all(d.get_fill_opacity() == 0 for d in m.layers[0])


def test_scale_shape_manual(df):
    sc = scale_shape_manual({"a": "star", "b": "plus", "c": "square_open"})
    m = ggplot(df, aes("x", "y", shape="g")).geom_point().add(sc).build()
    assert [type(d) for d in m.layers[0]][::2] == [Star, Cross, Square]


def test_scale_shape_manual_validates():
    with pytest.raises(ValueError, match="circle, triangle"):
        scale_shape_manual(["circle", "blob"])


def test_point_legend_glyphs_match_mapping(df):
    m = ggplot(df, aes("x", "y", shape="g", color="g")).geom_point().build()
    assert len(m.legends) == 1
    leg = m.legends[0]
    assert set(leg.aesthetics) == {"shape", "color"}
    assert [type(k[0]) for k in leg.keys] == [Dot, Triangle, Square]
    cols = [c.to_hex().upper() for c in m.scales["color"].map(["a", "b", "c"])]
    assert [k[0].get_fill_color().to_hex().upper() for k in leg.keys] == cols


def test_point_dot_radius_still_size(df):
    m = ggplot(df, aes("x", "y")).geom_point(size=0.1).build()
    assert all(d.radius == pytest.approx(0.1) for d in m.layers[0])


# -- linetype -----------------------------------------------------------------


def test_parse_linetype():
    assert parse_linetype(None) is None
    assert parse_linetype("solid") is None
    assert parse_linetype("dashed") == (4, 4)
    assert parse_linetype("dotdash") == parse_linetype("dashdot")
    assert parse_linetype(2) == (4, 4)
    assert parse_linetype("1343") == (1, 3, 4, 3)
    assert parse_linetype([2, 1]) == (2.0, 1.0)
    for lt in LINETYPES:
        parse_linetype(lt)
    with pytest.raises(ValueError, match="Valid linetypes"):
        parse_linetype("wavy")


def test_dash_polyline_lengths_follow_pattern():
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=float)  # length 2
    dashes = dash_polyline(pts, (0.3, 0.15))
    lens = [np.linalg.norm(np.diff(d, axis=0), axis=1).sum() for d in dashes]
    np.testing.assert_allclose(lens, [0.3, 0.3, 0.3, 0.3, 0.2])
    # the 3rd dash (0.9 .. 1.2) turns the corner at arc length 1
    assert [len(d) for d in dashes] == [2, 2, 3, 2, 2]


def test_dash_length_independent_of_path_length():
    short = path_mobject(np.array([[0, 0, 0], [1, 0, 0]], float), RED, 3.0, 1.0)
    long = path_mobject(np.array([[0, 0, 0], [4, 0, 0]], float), RED, 3.0, 1.0)
    ds, dl = apply_linetype(short, "dashed"), apply_linetype(long, "dashed")
    assert ds[0].get_end()[0] == pytest.approx(4 * DASH_UNIT * 3.0)
    assert dl[0].get_end()[0] == pytest.approx(4 * DASH_UNIT * 3.0)
    assert len(dl) > len(ds) > 1


def test_line_mapped_linetype(df):
    m = ggplot(df, aes("x", "y", linetype="g")).geom_line().build()
    marks = m.layers[0].marks
    assert len(marks) == 3
    assert not isinstance(marks[0], VGroup) or len(marks[0]) == 0  # solid: one path
    assert isinstance(marks[1], VGroup) and len(marks[1]) > 1  # dashed
    assert isinstance(marks[2], VGroup) and len(marks[2]) > 1  # dotted
    w = marks[1][0].get_stroke_width()
    assert w == pytest.approx(3.0)
    assert list(m.scales["linetype"].get_palette_values()) == LINETYPES[:3]


def test_line_constant_linetype(df):
    m = ggplot(df, aes("x", "y")).geom_line(linetype="longdash").build()
    assert len(m.layers[0].marks[0]) > 1


@pytest.mark.parametrize(
    "layer",
    [
        lambda: ggmanim.geom_vline(xintercept=3, linetype="dotted"),
        lambda: ggmanim.geom_hline(yintercept=3, linetype="dotted"),
        lambda: ggmanim.geom_function(fun=np.sin, linetype="dashed"),
        lambda: ggmanim.geom_smooth(linetype="dashed", se=False),
    ],
)
def test_other_line_geoms_take_linetype(df, layer):
    m = ggplot(df, aes("x", "y")).geom_point().add(layer()).build()
    mark = m.layers[1].marks[0]
    assert isinstance(mark, VGroup) and len(mark) > 1


def test_scale_linetype_manual(df):
    m = (
        ggplot(df, aes("x", "y", linetype="g"))
        .geom_line()
        .add(scale_linetype_manual(["twodash", "solid", "44"]))
        .build()
    )
    marks = m.layers[0].marks
    assert len(marks[0]) > 1 and len(marks[2]) > 1
    with pytest.raises(ValueError, match="Valid linetypes"):
        scale_linetype_manual(["solid", "zigzag"])


def test_scale_linetype_registered(df):
    m = ggplot(df, aes("x", "y", linetype="g")).geom_line().scale_linetype().build()
    assert m.scales["linetype"].get_palette_values() == LINETYPES[:3]
    assert scale_linetype().aesthetics == ("linetype",)
    assert scale_shape().aesthetics == ("shape",)


def test_line_legend_keys_dashed_and_fit(df):
    m = ggplot(df, aes("x", "y", linetype="g")).geom_line().build()
    keys = [k[0] for k in m.legends[0].keys]
    assert isinstance(keys[0], Line)
    for k in keys[1:]:
        assert len(k) >= 2  # at least two pattern repeats inside the key
        assert k.width <= KEY_SIZE + 1e-6


def test_draw_key_direct():
    assert isinstance(GeomPoint().draw_key({"shape": "star"}), Star)
    assert len(GeomLine().draw_key({"linetype": "dashed"})) > 1
    assert len(GeomVline().draw_key({"linetype": "dashed"})) > 1
    key = GeomSmooth().draw_key({"linetype": "dotted"})
    assert len(key[1]) > 1
