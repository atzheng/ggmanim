"""Theme elements (ggm-4a9.8): backgrounds, borders, grid split, per-axis
switches, rotated tick labels, text styles, title position, margins and the
complete theme set."""

import dataclasses

import manim
import numpy as np
import pandas as pd
import pytest
from manim import DOWN, LEFT, WHITE, ManimColor, config

import ggmanim
from ggmanim.build import fit_to_frame
from ggmanim.coords import coord_cartesian
from ggmanim.plot import ggplot
from ggmanim.registry import COMPONENTS
from ggmanim.scales import scale_x_continuous, scale_y_continuous
from ggmanim.theme import (
    Theme,
    theme,
    theme_bw,
    theme_classic,
    theme_dark,
    theme_default,
    theme_gray,
    theme_grey,
    theme_light,
    theme_linedraw,
    theme_manim,
    theme_minimal,
    theme_void,
)

COMPLETE = {
    "theme_default": theme_default,
    "theme_manim": theme_manim,
    "theme_gray": theme_gray,
    "theme_grey": theme_grey,
    "theme_bw": theme_bw,
    "theme_linedraw": theme_linedraw,
    "theme_light": theme_light,
    "theme_dark": theme_dark,
    "theme_minimal": theme_minimal,
    "theme_classic": theme_classic,
    "theme_void": theme_void,
}


def _scales(x=(0, 1000), y=(-2, 7)):
    sx, sy = scale_x_continuous(), scale_y_continuous()
    sx.train(list(x))
    sy.train(list(y))
    return sx, sy


def _panel(th, labels=None):
    sx, sy = _scales()
    return coord_cartesian().build(
        sx, sy, th, labels if labels is not None else {"x": "X", "y": "Y"}
    )


def _df():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 1.0, 4.0, 3.0], "g": list("abab")})


def _points(mob) -> np.ndarray:
    pts = mob.get_all_points()
    return pts - pts.mean(axis=0)


# -- complete themes ---------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(COMPLETE))
def test_complete_theme_registered_exported_and_builds(name):
    th = COMPLETE[name]()
    assert th.complete and name in COMPONENTS and name in ggmanim.__all__
    p = ggplot(_df(), ggmanim.aes("x", "y", color="g")).geom_point().labs(title="T")
    m = p.add(th).build()
    assert len(m.layers[0]) == 4
    if th.plot_background is not None:
        assert m.background is m.submobjects[0]
        for part in (m.panel, m.legend, m.title):
            assert m.background.get_left()[0] <= part.get_left()[0] + 1e-6
            assert m.background.get_top()[1] >= part.get_top()[1] - 1e-6
    else:
        assert m.background is None


def test_theme_manim_is_the_default_look():
    assert theme_manim() == theme_default()
    assert theme_manim().plot_background is None and theme_manim().foreground is None
    assert theme_grey() == theme_gray()


def test_ggplot_looks():
    gray = _panel(theme_gray())
    assert gray.background is gray.submobjects[0]
    assert gray.background.get_fill_color() == ManimColor("#EBEBEB")
    assert gray.grid[0].get_stroke_color() == ManimColor(WHITE)
    assert gray.axes.x_axis.get_stroke_opacity() == 0 and len(gray.x_ticks) > 0
    assert gray.border is None
    bw = _panel(theme_bw())
    assert bw.border is bw.submobjects[-1] and bw.border.get_fill_opacity() == 0
    assert bw.background.get_fill_color() == ManimColor(WHITE)
    np.testing.assert_allclose(
        bw.border.get_corner(DOWN + LEFT), bw.background.get_corner(DOWN + LEFT)
    )
    void = _panel(theme_void())
    assert len(void.grid) == len(void.x_ticks) == len(void.x_tick_labels) == 0
    assert void.x_title is None and void.y_title is None
    assert void.axes.x_axis.get_stroke_opacity() == 0 == void.axes.y_axis.get_stroke_opacity()
    light = _panel(theme_light())
    assert light.x_ticks[0].get_stroke_color() == ManimColor("#B3B3B3")


# -- per-axis switches --------------------------------------------------------------
def test_blanked_elements_absent_from_panel():
    full = _panel(theme_default())
    nx, ny = len(full.panel_params.x_breaks), len(full.panel_params.y_breaks)
    th = theme_default().merge(
        theme(axis_text_x=False, axis_ticks_y=False, axis_title_y=False, axis_line_x=False)
    )
    p = _panel(th)
    assert len(p.x_tick_labels) == 0 and len(p.y_tick_labels) == ny
    assert len(p.y_ticks) == 0 and len(p.x_ticks) == nx
    assert p.y_title is None and p.x_title is not None
    assert p.axes.x_axis.get_stroke_opacity() == 0 and p.axes.y_axis.get_stroke_opacity() == 1
    for m in (*p.x_tick_labels, *p.y_ticks):
        assert m not in p.submobjects
    # without ticks the y labels move in by the tick length
    gap = full.y_tick_labels[0].get_right()[0] - p.y_tick_labels[0].get_right()[0]
    assert gap == pytest.approx(-theme_default().tick_length)


def test_per_axis_switch_overrides_global():
    th = theme_default().merge(theme(tick_labels=False, axis_text_y=True))
    assert th.axis_element("axis_text", "y") and not th.axis_element("axis_text", "x")
    p = _panel(th)
    assert len(p.x_tick_labels) == 0 and len(p.y_tick_labels) > 0
    with pytest.raises(ValueError):
        th.axis_element("axis_text", "z")


def test_grid_major_per_axis():
    p = _panel(theme_default().merge(theme(panel_grid_major_x=False)))
    assert len(p.grid_major) == len(p.panel_params.y_breaks)
    assert all(abs(line.get_start()[1] - line.get_end()[1]) < 1e-9 for line in p.grid_major)


# -- rotated tick labels -------------------------------------------------------------
@pytest.mark.parametrize("angle", [30, 45, 90, -45])
def test_x_labels_rotated_and_below_axis(angle):
    plain = _panel(theme_default())
    p = _panel(theme_default().merge(theme(axis_text_x_angle=angle)))
    axis_y = p.axes.x_axis.get_start()[1]
    tick_bottom = p.x_ticks[0].get_bottom()[1]
    for lab, ref, tick in zip(p.x_tick_labels, plain.x_tick_labels, p.x_ticks):
        rotated = ref.copy().rotate(angle * manim.DEGREES)
        np.testing.assert_allclose(_points(lab), _points(rotated), atol=1e-6)
        assert lab.get_top()[1] <= tick_bottom - theme_default().tick_label_buff + 1e-6
        assert lab.get_top()[1] < axis_y
        # the label's end sits under its own tick
        end = lab.get_right() if angle > 0 else lab.get_left()
        assert abs(end[0] - tick.get_center()[0]) < max(lab.width, 0.05)
    # the x title is pushed below the (taller) rotated labels
    assert p.x_title.get_top()[1] < min(lab.get_bottom()[1] for lab in p.x_tick_labels)


def test_x_labels_explicit_just_pins_anchor():
    th = theme_default().merge(
        theme(axis_text_x_angle=90, axis_text_x_hjust=1, axis_text_x_vjust=0.5)
    )
    p = _panel(th)
    gap = th.tick_length + th.tick_label_buff
    for lab, tick in zip(p.x_tick_labels, p.x_ticks):
        # 90 degrees: the label's right end becomes its top, centred on the tick
        assert lab.get_top()[1] == pytest.approx(tick.get_start()[1] - gap)
        assert lab.get_center()[0] == pytest.approx(tick.get_center()[0])


def test_y_labels_rotated_left_of_axis():
    p = _panel(theme_default().merge(theme(axis_text_y_angle=90)))
    axis_x = p.axes.y_axis.get_start()[0]
    for lab, tick in zip(p.y_tick_labels, p.y_ticks):
        assert lab.get_right()[0] < axis_x
        assert lab.get_center()[1] == pytest.approx(tick.get_center()[1])
        assert lab.height > lab.width or len(lab.get_all_points()) == 0 or lab.width < 0.3


def test_angle_zero_matches_next_to():
    p = _panel(theme_default())
    gap = theme_default().tick_length + theme_default().tick_label_buff
    for lab, tick in zip(p.x_tick_labels, p.x_ticks):
        ref = lab.copy().next_to(tick.get_start(), DOWN, buff=gap)
        np.testing.assert_allclose(lab.get_center(), ref.get_center(), atol=1e-9)
    for lab, tick in zip(p.y_tick_labels, p.y_ticks):
        ref = lab.copy().next_to(tick.get_start(), LEFT, buff=gap)
        np.testing.assert_allclose(lab.get_center(), ref.get_center(), atol=1e-9)


# -- text styles -------------------------------------------------------------------
def test_text_colour_overrides():
    th = theme_default().merge(
        theme(axis_text_color="#FF0000", axis_title_color="#00FF00", title_color="#0000FF")
    )
    p = _panel(th)
    assert p.x_tick_labels[0].get_color() == ManimColor("#FF0000")
    assert p.x_title.get_color() == ManimColor("#00FF00")
    m = (
        ggplot(_df(), ggmanim.aes("x", "y"))
        .geom_point()
        .labs(title="T", caption="C")
        .add(th)
        .build()
    )
    assert m.title.get_color() == ManimColor("#0000FF")
    assert m.caption.get_color() == ManimColor("#FF0000")


def test_font_per_element(monkeypatch):
    calls = []

    class FakeText(manim.VMobject):
        def __init__(self, text, **kwargs):
            super().__init__()
            calls.append((text, kwargs.get("font")))
            self.set_points_as_corners([[0, 0, 0], [0.3, 0, 0], [0.3, 0.1, 0]])

    monkeypatch.setattr(manim, "Text", FakeText)
    th = theme(font="Base", axis_title_font="Serif")
    _panel(th, {"x": "the x title", "y": None})
    fonts = dict(calls)
    assert fonts["the x title"] == "Serif"
    assert all(f == "Base" for t, f in calls if t != "the x title")


def test_tex_engine_ignores_font(monkeypatch):
    seen = {}

    def fake_tex(text, **kwargs):
        seen.update(kwargs)
        return manim.Square()

    monkeypatch.setattr(manim, "Tex", fake_tex)
    theme(text_engine="tex", title_font="Serif").make_text(
        "a", **theme(title_font="Serif").text_style("title")
    )
    assert "font" not in seen


def test_plot_title_position():
    base = ggplot(_df(), ggmanim.aes("x", "y", color="g")).geom_point().labs(title="T", caption="C")
    m = base.build()
    assert m.title.get_left()[0] == pytest.approx(m.axes.get_left()[0])
    assert m.caption.get_right()[0] == pytest.approx(m.axes.get_right()[0])
    m2 = base.add(theme(plot_title_position="plot", legend_position="left")).build()
    assert m2.title.get_left()[0] == pytest.approx(
        min(m2.panel.get_left()[0], m2.legend.get_left()[0])
    )
    assert m2.title.get_left()[0] < m2.axes.get_left()[0]


# -- plot margin / background ----------------------------------------------------------
def test_plot_margin_normalised_and_fit():
    assert theme(plot_margin=0.5).plot_margin == (0.5, 0.5, 0.5, 0.5)
    assert theme(plot_margin=[1, 0, 0, 2]).plot_margin == (1.0, 0.0, 0.0, 2.0)
    sq = manim.Square(side_length=100)
    fit_to_frame(sq, margin=(1.0, 0.0, 0.0, 2.0))
    assert sq.get_top()[1] == pytest.approx(config.frame_height / 2 - 1.0)
    assert sq.get_left()[0] >= -config.frame_width / 2 + 2.0 - 1e-6
    assert sq.get_right()[0] <= config.frame_width / 2 + 1e-6


def test_plot_background_padded_by_margin():
    th = theme(plot_background="#FFFFFF", plot_margin=0.3, panel_width=4, panel_height=3)
    m = ggplot(_df(), ggmanim.aes("x", "y")).geom_point().add(th).build()
    bg = m.background
    assert bg is m.submobjects[0] and bg.get_fill_color() == ManimColor(WHITE)
    content = manim.VGroup(*m.submobjects[1:])
    assert content.get_left()[0] - bg.get_left()[0] == pytest.approx(0.3)
    assert bg.get_top()[1] - content.get_top()[1] == pytest.approx(0.3)


# -- composition and validation -------------------------------------------------------
def test_partial_merge_keeps_other_fields():
    merged = theme_bw().merge(theme(axis_text_x_angle=45, panel_grid_minor=False))
    assert merged.axis_text_x_angle == 45 and not merged.panel_grid_minor
    assert merged.panel_border and merged.panel_background == theme_bw().panel_background
    assert merged.plot_background == theme_bw().plot_background and merged.complete
    # and a later partial keeps the earlier one
    again = merged.merge(theme(axis_text_y=False))
    assert again.axis_text_x_angle == 45 and again.axis_text_y is False


def test_panel_grid_alias():
    t = theme(panel_grid=False)
    assert not t.panel_grid_major and not t.panel_grid_minor and not t.panel_grid
    assert theme(panel_grid=False, panel_grid_minor=True).panel_grid_minor
    merged = theme_default().merge(theme(panel_grid=False))
    assert not merged.panel_grid and len(_panel(merged).grid) == 0
    assert theme_default().replace(panel_grid=False).panel_grid_minor is False
    assert "panel_grid" not in {f.name for f in dataclasses.fields(Theme)}


@pytest.mark.parametrize(
    "bad",
    [
        {"plot_title_position": "left"},
        {"panel_background": "not-a-colour"},
        {"grid_color": 12.5j},
        {"axis_text_x_hjust": 1.5},
        {"axis_text_y_vjust": -0.1},
        {"grid_opacity": 2},
        {"axis_text_x_angle": float("nan")},
        {"axis_text_y_angle": "45"},
        {"plot_margin": (1, 2)},
        {"plot_margin": -1},
    ],
)
def test_validation(bad):
    with pytest.raises(ValueError):
        theme(**bad)


def test_documented():
    doc = Theme.__doc__ or ""
    for f in dataclasses.fields(Theme):
        if f.name not in ("complete", "overrides"):
            assert f.name in doc, f.name
