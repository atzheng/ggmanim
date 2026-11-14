"""geom_text / geom_label: placement (hjust/vjust/angle/nudge), size, check_overlap,
after_stat labels, parse, and the shared direct-label helper."""

import shutil
import warnings

import numpy as np
import pandas as pd
import pytest
from manim import RED, MathTex, Mobject, VGroup

import ggmanim
from ggmanim import aes, after_stat, geom_col, geom_label, geom_line, geom_text, ggplot, theme
from ggmanim.coords import CoordCartesian
from ggmanim.geoms import GeomLabel, GeomText
from ggmanim.geoms._labels import (
    direct_label,
    font_size_for,
    format_label,
    place_text,
    resolve_just,
)
from ggmanim.layer import Layer
from ggmanim.registry import COMPONENTS
from ggmanim.scales import scale_x_continuous, scale_y_continuous
from ggmanim.theme import theme_default

DF = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 4.0, 9.0], "name": ["a", "bb", "ccc"]})


def _panel(x=(0, 10), y=(0, 10)):
    sx, sy = scale_x_continuous(), scale_y_continuous()
    sx.train(np.asarray(x, dtype=float))
    sy.train(np.asarray(y, dtype=float))
    return CoordCartesian().build(sx, sy, theme_default())


def _draw(geom, data, panel=None, **params):
    panel = panel if panel is not None else _panel()
    data = data.assign(group=-1) if "group" not in data.columns else data
    return geom.draw_panel(data, panel.panel_params, panel, theme_default(), **params), panel


def _one(**cols):
    return pd.DataFrame({"x": [5.0], "y": [5.0], "label": ["Hello"], **cols})


def test_registered_and_exported():
    for n in ("geom_text", "geom_label", "annotate"):
        assert n in ggmanim.__all__
    assert "geom_text" in COMPONENTS and "geom_label" in COMPONENTS
    layer = geom_text(aes(label="name"), nudge_y=1)
    assert isinstance(layer, Layer) and isinstance(layer.geom, GeomText)
    assert layer.params["nudge_y"] == 1
    assert isinstance(geom_label().geom, GeomLabel)
    assert hasattr(ggplot(DF, aes("x", "y")), "geom_text")


@pytest.mark.parametrize("hjust,vjust", [(0, 0), (1, 1), (0.5, 0.5), (0, 1), (1, 0)])
def test_label_anchored_at_c2p_with_justification(hjust, vjust):
    out, panel = _draw(GeomText(), _one(hjust=[hjust], vjust=[vjust]))
    (mob,) = out.marks
    p = panel.transform(5.0, 5.0)[0]
    anchor = np.array(
        [mob.get_left()[0] + hjust * mob.width, mob.get_bottom()[1] + vjust * mob.height]
    )
    np.testing.assert_allclose(anchor, p[:2], atol=1e-6)


def test_named_justification_and_constant_params():
    out, panel = _draw(GeomText(), _one(), hjust="left", vjust="top")
    (mob,) = out.marks
    p = panel.transform(5.0, 5.0)[0]
    np.testing.assert_allclose([mob.get_left()[0], mob.get_top()[1]], p[:2], atol=1e-6)


def test_inward_outward():
    data = pd.DataFrame({"x": [1.0, 9.0], "y": [5.0, 5.0], "label": ["L", "R"]})
    out, panel = _draw(GeomText(), data, hjust="inward")
    left, right = out.marks
    pl, pr = panel.transform([1.0, 9.0], [5.0, 5.0])
    assert left.get_left()[0] == pytest.approx(pl[0])  # pushed right, towards the centre
    assert right.get_right()[0] == pytest.approx(pr[0])
    out, _ = _draw(GeomText(), data, hjust="outward")
    assert out.marks[0].get_right()[0] == pytest.approx(pl[0])
    assert resolve_just("inward", 0.5) == 0.5
    with pytest.raises(ValueError):
        resolve_just("sideways")


def test_angle_rotates_about_anchor():
    flat, panel = _draw(GeomText(), _one(hjust=[0.0], vjust=[0.0]))
    rot, _ = _draw(GeomText(), _one(hjust=[0.0], vjust=[0.0], angle=[90.0]), panel)
    p = panel.transform(5.0, 5.0)[0]
    # rotated 90 degrees about the bottom-left corner: now extends up and to the left
    assert rot.marks[0].height == pytest.approx(flat.marks[0].width, rel=1e-6)
    assert rot.marks[0].get_bottom()[1] == pytest.approx(p[1])
    assert rot.marks[0].get_right()[0] == pytest.approx(p[0])


def test_size_scales_relative_to_theme():
    t = theme_default()
    assert font_size_for(t, 3.88) == pytest.approx(t.axis_text_size)
    assert font_size_for(t, 7.76) == pytest.approx(2 * t.axis_text_size)
    small, _ = _draw(GeomText(), _one())
    big, _ = _draw(GeomText(), _one(size=[7.76]))
    assert big.marks[0].height == pytest.approx(2 * small.marks[0].height, rel=0.05)


def test_colour_alpha_and_empty_labels_skipped():
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "label": ["a", "", None]})
    out, _ = _draw(GeomText(), data, color=RED, alpha=0.5)
    assert len(out.marks) == 1 and out.keys == [0]
    m = out.marks[0]
    assert m.get_fill_color().to_hex().upper() == RED.to_hex().upper()
    assert m.get_fill_opacity() == pytest.approx(0.5)


def test_fontface_and_family_build():
    out, _ = _draw(GeomText(), _one(fontface=["bold.italic"]), family="Sans")
    assert len(out.marks) == 1


def test_check_overlap_greedy_in_data_order():
    data = pd.DataFrame(
        {"x": [5.0, 5.05, 9.0], "y": [5.0, 5.0, 1.0], "label": ["first", "second", "far"]}
    )
    out, _ = _draw(GeomText(), data, check_overlap=True)
    assert out.keys == [0, 2]
    out, _ = _draw(GeomText(), data)
    assert out.keys == [0, 1, 2]


def test_nudge_moves_labels_and_trains_scales():
    m = ggplot(DF, aes("x", "y", label="name")).add(geom_text(nudge_x=0.5, nudge_y=-1)).build()
    ld = m.layer_data[0]
    np.testing.assert_allclose(ld["x"], DF["x"] + 0.5)
    np.testing.assert_allclose(ld["y"], DF["y"] - 1)
    mob = m.layers[0]
    p = m.panel.transform(3.5, 8.0)[0]
    np.testing.assert_allclose(mob.marks[2].get_center()[:2], p[:2], atol=1e-6)
    assert m.panel.panel_params.x_range[1] >= 3.5


def test_after_stat_label_formats_numbers():
    m = ggplot(DF, aes("x", "y")).add(geom_text(aes(label=after_stat("y")))).build()
    texts = [t.text for t in m.layers[0].marks]
    assert texts == ["1", "4", "9"]
    assert format_label(2.5) == "2.5" and format_label(float("nan")) is None


def test_counts_on_bars_identity():
    counts = pd.DataFrame({"cut": ["a", "b", "c"], "n": [3, 7, 5]})
    m = (
        ggplot(counts, aes("cut", "n"))
        .add(geom_col())
        .add(geom_text(aes(label="n"), vjust=0, nudge_y=0.2))
        .build()
    )
    bars, labels = m.layers[0].marks, m.layers[1].marks
    for bar, lab in zip(bars, labels):
        assert lab.get_bottom()[1] > bar.get_top()[1]
        assert lab.get_center()[0] == pytest.approx(bar.get_center()[0], abs=1e-6)


def test_counts_on_bars_after_stat_count():
    df = pd.DataFrame({"cut": list("aabbbc")})
    m = (
        ggplot(df, aes("cut"))
        .add(geom_text(aes(label=after_stat("count")), stat="count", vjust=0))
        .build()
    )
    assert [t.text for t in m.layers[0].marks] == ["2", "3", "1"]


def test_geom_label_box_contains_text():
    out, panel = _draw(GeomLabel(), _one(), fill="#FFEE00", label_padding=0.1, label_r=0.0)
    (mark,) = out.marks
    assert isinstance(mark, VGroup) and len(mark) == 2
    box, text = mark
    assert box.width == pytest.approx(text.width + 0.2)
    assert box.get_fill_color().to_hex().upper() == "#FFEE00"
    np.testing.assert_allclose(mark.get_center(), panel.transform(5.0, 5.0)[0], atol=1e-6)
    out, _ = _draw(GeomLabel(), _one(hjust=[0.0]), label_size=0)
    assert out.marks[0][0].get_stroke_width() == 0


def test_geom_label_default_colours_contrast():
    hexes = lambda m: (m[0].get_fill_color().to_hex(), m[1].get_fill_color().to_hex())  # noqa: E731
    out, _ = _draw(GeomLabel(), _one())  # dark default theme: white text on black
    assert hexes(out.marks[0]) == ("#000000", "#FFFFFF")
    out, _ = _draw(GeomLabel(), _one(), fill="#FFFFFF")  # light fill -> black text
    assert hexes(out.marks[0]) == ("#FFFFFF", "#000000")
    out, _ = _draw(GeomLabel(), _one(), color="#FFFF00")
    assert hexes(out.marks[0]) == ("#000000", "#FFFF00")


def test_draw_keys():
    for g in (GeomText(), GeomLabel()):
        key = g.draw_key({"color": RED})
        assert isinstance(key, Mobject) and key.height > 0


def test_parse_without_tex_warns_and_draws_text():
    with pytest.warns(UserWarning, match="parse=True"):
        out, _ = _draw(GeomText(), _one(), parse=True)
    assert len(out.marks) == 1


@pytest.mark.skipif(shutil.which("latex") is None, reason="LaTeX not installed")
def test_parse_renders_mathtex_with_tex_engine():
    data = pd.DataFrame({"x": [1.0], "y": [1.0]})
    m = (
        ggplot(data, aes("x", "y"))
        .add(geom_text(label=r"\alpha^2", parse=True))
        .add(theme(text_engine="tex"))
        .build()
    )
    assert isinstance(m.layers[0].marks[0], MathTex)


def test_place_text_and_direct_label_helpers():
    t = theme_default()
    lab = direct_label(t, "end", RED, np.array([1.0, 2.0, 0.0]), nudge=(0.1, 0.0))
    assert lab.get_left()[0] == pytest.approx(1.1)
    assert lab.get_center()[1] == pytest.approx(2.0)
    assert direct_label(t, "", RED, np.zeros(3)) is None
    mob = t.make_text("x")
    place_text(mob, np.zeros(3), 1.0, 1.0)
    np.testing.assert_allclose(mob.get_corner([1, 1, 0])[:2], [0, 0], atol=1e-9)


def test_line_direct_label_placed_after_last_point():
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 0.5]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = ggplot(df, aes("x", "y")).add(geom_line(label="end")).build()
    lab = m.layers[0].labels[0]
    p = m.panel.transform(2.0, 0.5)[0]
    assert lab.get_left()[0] == pytest.approx(p[0] + 0.1)
    assert lab.get_center()[1] == pytest.approx(p[1])


def test_vline_hline_direct_labels():
    df = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 10.0]})
    m = (
        ggplot(df, aes("x", "y"))
        .geom_vline(xintercept=5, label="vline")
        .geom_hline(yintercept=5, label="h")
        .build()
    )
    pp = m.panel.panel_params
    top = m.panel.transform(5.0, pp.y_range[1])[0]
    v = m.layers[0].labels[0]
    assert v.get_right()[0] == pytest.approx(top[0] - 0.05)
    assert v.get_top()[1] == pytest.approx(top[1])
    assert v.height > v.width  # rotated
    right = m.panel.transform(pp.x_range[1], 5.0)[0]
    h = m.layers[1].labels[0]
    assert h.get_right()[0] == pytest.approx(right[0])
    assert h.get_bottom()[1] == pytest.approx(right[1] + 0.05)
