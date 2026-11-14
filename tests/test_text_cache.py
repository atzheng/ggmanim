"""Text memoisation in Theme.make_text and single-mobject grids (ggm-bam.2)."""

import manim
import numpy as np
import pandas as pd
import pytest
from manim import Line, tempconfig

from ggmanim import aes, geom_point, ggplot
from ggmanim._text_cache import (
    clear_text_cache,
    fast_copy,
    text_cache_info,
    text_key,
)
from ggmanim.coords import GridLines, coord_cartesian, grid_line_count
from ggmanim.scales import scale_x_continuous, scale_y_continuous
from ggmanim.theme import theme, theme_default


def _family_equal(a: manim.Mobject, b: manim.Mobject) -> None:
    fa, fb = a.get_family(), b.get_family()
    assert len(fa) == len(fb)
    for x, y in zip(fa, fb):
        assert type(x) is type(y)
        np.testing.assert_array_equal(x.points, y.points)
        if isinstance(x, manim.VMobject):
            np.testing.assert_array_equal(x.fill_rgbas, y.fill_rgbas)
            np.testing.assert_array_equal(x.stroke_rgbas, y.stroke_rgbas)


def test_repeat_request_is_a_fresh_copy():
    clear_text_cache()
    th = theme_default()
    a = th.make_text("12.5", font_size=24, color="#ff0000")
    b = th.make_text("12.5", font_size=24, color="#ff0000")
    assert a is not b
    assert text_cache_info().hits == 1 and text_cache_info().misses == 1
    _family_equal(a, b)
    assert not np.shares_memory(a.submobjects[0].points, b.submobjects[0].points)
    # mutating what a caller got leaves later copies untouched
    a.shift(3 * manim.RIGHT).set_color("#00ff00").rotate(1.0)
    c = th.make_text("12.5", font_size=24, color="#ff0000")
    _family_equal(b, c)


def test_cached_text_matches_an_uncached_build():
    clear_text_cache()
    th = theme_default()
    th.make_text("level 03", font_size=24, color="#123456")
    got = th.make_text("level 03", font_size=24, color="#123456")
    _family_equal(got, manim.Text("level 03", font_size=24, color="#123456"))
    assert got.original_text == "level 03"
    para = th.make_text("a\nbb", font_size=20, color="#ffffff", align="right")
    again = th.make_text("a\nbb", font_size=20, color="#ffffff", align="right")
    assert isinstance(again, manim.Paragraph)
    _family_equal(para, again)


def test_key_distinguishes_what_changes_the_result():
    base = text_key("text", "x", 24, "#ff0000", "", {})
    assert base == text_key("text", "x", 24.0, manim.ManimColor("#FF0000"), "", {})
    for other in (
        text_key("tex", "x", 24, "#ff0000", "", {}),
        text_key("text", "y", 24, "#ff0000", "", {}),
        text_key("text", "x", 25, "#ff0000", "", {}),
        text_key("text", "x", 24, "#ff0001", "", {}),
        text_key("text", "x", 24, "#ff0000", "right", {}),
        text_key("text", "x", 24, "#ff0000", "", {"font": "Serif"}),
        text_key("text", "x", 24, "#ff0000", "", {"weight": "BOLD"}),
    ):
        assert other != base
    # unhashable keyword arguments are built uncached
    assert text_key("text", "x", 24, "#ff0000", "", {"t2c": {"x": "#ff0000"}}) is None


def test_key_follows_tex_template():
    key = text_key("tex", "x", 24, "#ff0000", "", {})
    template = manim.TexTemplate()
    template.add_to_preamble(r"\usepackage{mathrsfs}")
    with tempconfig({"tex_template": template}):
        assert text_key("tex", "x", 24, "#ff0000", "", {}) != key


def test_theme_font_and_style_enter_the_key():
    clear_text_cache()
    a = theme_default().make_text("abc", font_size=24)
    b = theme_default().merge(theme(font="Serif")).make_text("abc", font_size=24)
    assert text_cache_info().misses == 2
    assert b.font == "Serif" and a.font != "Serif"


def test_uncachable_and_passthrough_requests():
    th = theme_default()
    m = manim.Square()
    assert th.make_text(m) is m
    got = th.make_text("ab", t2c={"a": "#ff0000"})
    assert isinstance(got, manim.Text)
    for _ in range(2):  # errors are not cached
        with pytest.raises(ValueError, match="align"):
            th.make_text("a\nb", align="middle")


def test_fast_copy_is_a_deep_copy():
    text = manim.Text("Wq", font_size=30, color="#abcdef")
    dup = fast_copy(text)
    _family_equal(dup, text.copy())
    # shared references inside the family stay shared, between the clones
    assert dup.chars[0] is dup.submobjects[0]
    assert dup.submobjects[0] is not text.submobjects[0]
    dup.submobjects[0].shift(manim.UP)
    assert not np.array_equal(dup.submobjects[0].points, text.submobjects[0].points)


def test_second_build_of_a_plot_lays_out_no_text():
    df = pd.DataFrame({"x": np.linspace(0, 1, 100), "y": np.linspace(0, 1, 100) ** 2})
    plot = ggplot(df, aes("x", "y")).add(geom_point())
    clear_text_cache()
    plot.build()
    first = text_cache_info()
    assert first.misses > 0
    plot.build()
    second = text_cache_info()
    assert second.misses == first.misses and second.hits > first.hits


# -------------------------------------------------------------------------- grids
def _panel(th=None):
    sx, sy = scale_x_continuous(), scale_y_continuous()
    sx.train([0, 10])
    sy.train([0, 5])
    return coord_cartesian().build(sx, sy, th or theme_default(), {"x": "X", "y": "Y"})


def test_grid_is_one_mobject_per_axis():
    p = _panel()
    pp = p.panel_params
    assert all(isinstance(m, GridLines) for m in p.grid)
    assert len(p.grid_major) == 2 and len(p.grid_minor) == 2
    vertical, horizontal = p.grid_major
    assert vertical.n_lines == len(pp.x_breaks) and horizontal.n_lines == len(pp.y_breaks)
    assert len(vertical.get_subpaths()) == len(pp.x_breaks)
    assert grid_line_count(p.grid_minor) == len(pp.x_minor_breaks) + len(pp.y_minor_breaks)
    assert list(p.grid.submobjects) == [*p.grid_minor, *p.grid_major]


def test_grid_lines_have_the_points_of_lines():
    starts = np.array([[0.0, -1.0, 0.0], [0.7, -1.0, 0.0]])
    ends = np.array([[0.0, 2.0, 0.0], [0.7, 2.0, 0.0]])
    g = GridLines(starts, ends, stroke_width=2, stroke_color="#ff0000", stroke_opacity=0.5)
    expected = np.concatenate([Line(a, b).points for a, b in zip(starts, ends)])
    np.testing.assert_array_equal(g.points, expected)
    np.testing.assert_array_equal(g.starts, starts)
    np.testing.assert_array_equal(g.ends, ends)
    assert g.get_stroke_width() == 2 and g.get_stroke_opacity() == pytest.approx(0.5)
    assert GridLines(np.zeros((0, 3)), np.zeros((0, 3))).n_lines == 0


def test_switched_off_axes_have_no_grid_mobject():
    p = _panel(theme_default().merge(theme(panel_grid_major_x=False, panel_grid_minor=False)))
    assert len(p.grid_minor) == 0 and len(p.grid_major) == 1
