import dataclasses

import manim
import pytest
from manim import BLACK, WHITE, ManimColor, Square, tempconfig

from ggmanim.plot import ggplot
from ggmanim.registry import COMPONENTS
from ggmanim.theme import (
    foreground_for_background,
    theme,
    theme_classic,
    theme_default,
    theme_minimal,
)


def test_registered():
    for name in ("theme_default", "theme_minimal", "theme_classic"):
        assert name in COMPONENTS
    # ``plot.theme`` is the theme field, so the partial constructor can't be chaining sugar
    assert "theme" not in COMPONENTS


def test_complete_flags():
    assert theme_default().complete
    assert theme_minimal().complete
    assert theme_classic().complete
    assert not theme(legend_position="bottom").complete


def test_theme_is_immutable():
    t = theme_default()
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.axis_text_size = 3  # type: ignore[misc]


def test_theme_validation():
    with pytest.raises(TypeError):
        theme(not_a_field=1)
    with pytest.raises(ValueError):
        theme(text_engine="mathjax")
    with pytest.raises(ValueError):
        theme(legend_position="middle")


def test_merge_partial_overrides_only_given_fields():
    base = theme_minimal()
    merged = base.merge(theme(legend_position="bottom", axis_text_size=10))
    assert merged.legend_position == "bottom"
    assert merged.axis_text_size == 10
    assert merged.axis_line is False  # kept from minimal
    assert merged.ticks is False
    assert merged.complete  # still a complete theme
    assert base.legend_position == "right"  # base untouched


def test_merge_complete_replaces():
    assert theme(axis_text_size=10).merge(theme_classic()) == theme_classic()


def test_plot_composition():
    p = ggplot().add(theme_minimal(), theme(legend_position="none"))
    assert p.theme.legend_position == "none"
    assert p.theme.panel_grid and not p.theme.axis_line
    # a complete theme replaces earlier overrides
    p2 = p.add(theme_classic())
    assert p2.theme == theme_classic()
    # partial theme on a plot without theme: default + overrides
    p3 = ggplot().add(theme(axis_text_size=11))
    assert p3.theme.axis_text_size == 11
    assert p3.theme.axis_line is True
    # chaining sugar
    assert ggplot().theme_minimal().theme.axis_line is False


@pytest.mark.parametrize(
    "background, expected",
    [
        (WHITE, BLACK),
        ("#FFFFFF", BLACK),
        ("#EEEEEE", BLACK),
        (BLACK, WHITE),
        ("#000000", WHITE),
        ("#333333", WHITE),
    ],
)
def test_foreground_for_background(background, expected):
    assert foreground_for_background(background) == ManimColor(expected)


def test_foreground_follows_config_at_call_time():
    t = theme_default()  # created before the config change
    with tempconfig({"background_color": WHITE}):
        assert t.get_foreground() == ManimColor(BLACK)
        assert t.make_text("a").get_color() == ManimColor(BLACK)
    with tempconfig({"background_color": BLACK}):
        assert t.get_foreground() == ManimColor(WHITE)
        assert t.make_text("a").get_color() == ManimColor(WHITE)


def test_explicit_foreground_wins():
    t = theme_default().merge(theme(foreground="#FF0000"))
    with tempconfig({"background_color": WHITE}):
        assert t.get_foreground() == ManimColor("#FF0000")


def test_make_text_returns_mobject_unchanged():
    sq = Square()
    before = sq.get_center().copy()
    assert theme_default().make_text(sq) is sq
    assert (sq.get_center() == before).all()


def test_make_text_text_engine(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("LaTeX must not be used with text_engine='text'")

    monkeypatch.setattr(manim, "Tex", boom)
    monkeypatch.setattr(manim, "MathTex", boom)
    m = theme_default().make_text("hello", font_size=20)
    assert isinstance(m, manim.Text)
    assert m.text == "hello"


def test_make_text_tex_engine_dispatches_to_tex(monkeypatch):
    calls = []

    def fake_tex(text, **kwargs):
        calls.append((text, kwargs))
        return Square()

    monkeypatch.setattr(manim, "Tex", fake_tex)
    theme(text_engine="tex").make_text("x", font_size=12)
    assert calls and calls[0][0] == "x" and calls[0][1]["font_size"] == 12


def test_panel_size_fits_frame():
    t = theme_default()
    w, h = t.panel_size()
    wl, hl = t.panel_size(has_legend=True)
    assert 0 < w <= manim.config.frame_width and 0 < h <= manim.config.frame_height
    assert wl < w and hl == h
    assert theme(legend_position="none").panel_size(True) == (w, h)
    assert theme(panel_width=4, panel_height=3).panel_size(True) == (4.0, 3.0)
