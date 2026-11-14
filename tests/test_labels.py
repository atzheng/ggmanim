import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

import ggmanim
from ggmanim.coords import coord_cartesian, grid_line_count
from ggmanim.labels import (
    breaks_extended,
    breaks_log,
    breaks_pretty,
    breaks_width,
    label_comma,
    label_currency,
    label_date,
    label_dollar,
    label_math,
    label_number,
    label_percent,
    label_scientific,
    label_wrap,
    math_to_unicode,
    render_label,
)
from ggmanim.scales import scale_x_continuous, scale_x_log10, scale_y_continuous
from ggmanim.scales.breaks import (
    extended,
    log_breaks,
    log_minor_breaks,
    pretty,
    regular_minor_breaks,
    width_breaks,
)
from ggmanim.theme import theme, theme_classic, theme_default

NAN, INF = float("nan"), float("inf")
EDGE = [0, -1234.5, 1e-7, 12345678.9, NAN]


# --------------------------------------------------------------------------- formatters
def test_label_number_defaults_and_edge_cases():
    assert label_number()([0, 1234.5, -12345678.9]) == ["0.0", "1,234.5", "-12,345,678.9"]
    assert label_number()([1e-7]) == ["0.0000001"]
    assert label_number()([0]) == ["0"]
    assert label_number()([NAN, INF, -INF]) == ["", "Inf", "-Inf"]
    assert label_number()([-0.0, 5]) == ["0", "5"]


def test_label_number_caps_significant_digits():
    assert label_number()([0.9999999999, 2.0000000001]) == ["1", "2"]
    assert label_number()([-1.23456789, 3.3]) == ["-1.234568", "3.300000"]
    assert label_number()([12345678.123456]) == ["12,345,678"]  # integer digits kept


def test_label_number_accuracy_scale_marks():
    assert label_number(accuracy=0.01)([1, 2.346]) == ["1.00", "2.35"]
    assert label_number(accuracy=10)([1234, 5]) == ["1,230", "0"]
    assert label_number(accuracy=0.01)([-0.001]) == ["0.00"]  # no "-0.00"
    assert label_number(scale=1e-3, suffix="k")([1500, 20000]) == ["1.5k", "20.0k"]
    assert label_number(big_mark=" ", decimal_mark=",")([1234.5]) == ["1 234,5"]
    assert label_number(big_mark="")([1234567]) == ["1234567"]
    assert label_number(trim=False)([1, 1000]) == ["    1", "1,000"]
    with pytest.raises(ValueError):
        label_number(accuracy=0)([1])


def test_label_comma_percent():
    assert label_comma(accuracy=1)(EDGE) == ["0", "-1,234", "0", "12,345,679", ""]
    assert label_percent()([0, 0.25, -0.5, 1, NAN]) == ["0%", "25%", "-50%", "100%", ""]
    assert label_percent(accuracy=0.1)([0.1234]) == ["12.3%"]


def test_label_dollar():
    assert label_dollar is label_currency
    assert label_dollar()([0, 1500, -20]) == ["$0", "$1,500", "-$20"]
    assert label_dollar()([0.5, 2]) == ["$0.50", "$2.00"]
    assert label_dollar()([0.5, 2e5]) == ["$0", "$200,000"]  # too large for cents
    assert label_currency(prefix="", suffix=" €", accuracy=1)([1234]) == ["1,234 €"]
    assert label_dollar()([NAN]) == [""]


def test_label_scientific():
    assert label_scientific()(EDGE) == ["0.00e+00", "-1.23e+03", "1.00e-07", "1.23e+07", ""]
    assert label_scientific()([1e5, 2.5e5]) == ["1.0e+05", "2.5e+05"]
    assert label_scientific(digits=1)([123456]) == ["1e+05"]
    assert label_scientific()([1e-300]) == ["1e-300"]
    with pytest.raises(ValueError):
        label_scientific(0)


def test_label_math():
    f = label_math()
    assert f([1, 10, 1000, 0.001]) == ["$10^{0}$", "$10^{1}$", "$10^{3}$", "$10^{-3}$"]
    assert f([0, -5, NAN]) == ["", "", ""]
    assert f([2]) == ["$10^{0.301}$"]
    assert label_math(base=2)([8]) == ["$2^{3}$"]
    assert label_math(base=math.e)([math.e**2]) == ["$e^{2}$"]
    with pytest.raises(ValueError):
        label_math(base=1)


def test_math_rendering_per_engine():
    assert math_to_unicode("$10^{-3}$") == "10⁻³"
    assert math_to_unicode("$10^{12}$") == "10¹²"
    assert math_to_unicode("$2 \\times 10^5$") == "2 × 10⁵"
    assert math_to_unicode("$10^{0.301}$") == "10^(0.301)"  # no superscript '.'
    assert render_label("$10^{3}$", "text") == "10³"
    assert render_label("$10^{3}$", "tex") == "$10^{3}$"
    # non-math labels: unchanged for Text, LaTeX specials escaped for Tex
    assert render_label("$1,000", "text") == "$1,000"
    assert render_label("$1,000", "tex") == "\\$1,000"
    assert render_label("50%", "tex") == "50\\%"
    assert render_label("a_b & c", "tex") == "a\\_b \\& c"
    assert render_label("a\nb", "tex") == "a\\\\b"


def test_label_date():
    f = label_date()
    assert f([0, 19000]) == ["1970-01-01", "2022-01-08"]
    assert f(np.array(["2024-02-03", "NaT"], dtype="datetime64[D]")) == ["2024-02-03", ""]
    assert label_date("%b %Y")([dt.date(2020, 3, 1)]) == ["Mar 2020"]
    assert label_date("%H:%M", unit="s")([3600 * 13 + 60 * 5]) == ["13:05"]
    assert f([NAN]) == [""]
    with pytest.raises(ValueError):
        label_date(unit="weeks")


def test_label_wrap():
    assert label_wrap(10)(["a long category name", "x", ""]) == ["a long\ncategory\nname", "x", ""]


# --------------------------------------------------------------------------- breaks
@pytest.mark.parametrize(
    "lo, hi, m, expected",
    [
        # labeling::extended() outputs (Talbot, Lin & Hanrahan 2010 examples / ggplot2 defaults)
        (0, 100, 5, [0, 25, 50, 75, 100]),
        (1, 10, 5, [0, 2.5, 5, 7.5, 10]),
        (8.1, 14.1, 4, [8, 10, 12, 14]),
        (10.4, 33.9, 5, [10, 15, 20, 25, 30, 35]),
        (1.513, 5.424, 5, [2, 3, 4, 5]),
        (0, 1, 5, [0, 0.25, 0.5, 0.75, 1]),
        (-31.4, 68.1, 5, [-25, 0, 25, 50, 75]),
    ],
)
def test_extended_known_cases(lo, hi, m, expected):
    np.testing.assert_allclose(extended(lo, hi, m), expected, atol=1e-12)


def test_extended_vs_pretty():
    """Same "nice" family as pretty(), similar counts, tighter (not forced to cover)."""
    rng = np.random.default_rng(1)
    for _ in range(150):
        lo, hi = sorted(rng.uniform(-1e3, 1e3, 2))
        e, p = extended(lo, hi), pretty(lo, hi)
        assert 2 <= len(e) <= 10
        step = np.diff(e)
        assert np.allclose(step, step[0])
        mant = step[0] / 10 ** np.floor(np.log10(step[0]) + 1e-12)
        assert any(np.isclose(mant, q) for q in (1, 5, 2, 2.5, 4, 3, 10))
        # spans roughly the data, never wildly more than pretty's covering span
        assert e[-1] - e[0] <= (p[-1] - p[0]) * 1.5 + 1e-9
        assert e[0] <= lo + (hi - lo) * 0.5 and e[-1] >= hi - (hi - lo) * 0.5


def test_extended_degenerate_and_options():
    np.testing.assert_allclose(extended(5, 5, 3), [5, 5, 5])
    np.testing.assert_allclose(extended(10, 0), extended(0, 10))
    b = extended(1.513, 5.424, only_loose=True)
    assert b[0] <= 1.513 and b[-1] >= 5.424
    with pytest.raises(ValueError):
        extended(0, np.inf)


def test_break_helpers():
    np.testing.assert_allclose(breaks_pretty()((0, 1)), pretty(0, 1))
    np.testing.assert_allclose(breaks_pretty(n=10)((0, 1)), pretty(0, 1, 10))
    np.testing.assert_allclose(breaks_extended()((1, 10)), [0, 2.5, 5, 7.5, 10])
    np.testing.assert_allclose(breaks_width(5)((3, 27)), [0, 5, 10, 15, 20, 25, 30])
    np.testing.assert_allclose(breaks_width(0.5, offset=0.25)((0, 1)), [0.25, 0.75, 1.25])
    np.testing.assert_allclose(width_breaks(0.1, 0.35, 0.1), [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(breaks_log()((3, 2500)), [1, 10, 100, 1000, 1e4])
    np.testing.assert_allclose(breaks_log(base=2)((1, 16)), [1, 2, 4, 8, 16])
    with pytest.raises(ValueError):
        breaks_width(0)((0, 1))


def test_helpers_as_scale_arguments():
    s = scale_x_continuous(breaks=breaks_width(2.5), labels=label_number(accuracy=0.1))
    s.train([0, 10])
    np.testing.assert_allclose(s.get_breaks(), [0, 2.5, 5, 7.5, 10])
    assert s.get_labels() == ["0.0", "2.5", "5.0", "7.5", "10.0"]
    s = scale_x_log10(labels=label_math())
    s.train([3, 2500])
    assert s.get_labels() == ["$10^{1}$", "$10^{2}$", "$10^{3}$"]


def test_public_api():
    for name in ("label_number", "label_math", "breaks_extended", "breaks_width", "label_dollar"):
        assert name in ggmanim.__all__ and hasattr(ggmanim, name)


# --------------------------------------------------------------------------- minor breaks
def test_regular_and_log_minor_breaks():
    np.testing.assert_allclose(regular_minor_breaks([0, 1, 2], (-0.6, 2.3)), [-0.5, 0.5, 1.5])
    np.testing.assert_allclose(
        regular_minor_breaks([0, 2, 4], (0, 4), n=4), [0.5, 1, 1.5, 2.5, 3, 3.5]
    )
    assert len(regular_minor_breaks([1], (0, 2))) == 0
    m = log_minor_breaks([0, 1, 2], (0, 2))
    assert len(m) == 16  # 2..9 in each of two decades
    np.testing.assert_allclose(10 ** m[:8], [2, 3, 4, 5, 6, 7, 8, 9])


def test_scale_minor_breaks():
    s = scale_x_continuous()
    s.train([0, 10])
    np.testing.assert_allclose(s.get_minor_breaks(), [1, 3, 5, 7, 9])
    np.testing.assert_allclose(s.get_minor_breaks((-1, 11)), [-1, 1, 3, 5, 7, 9, 11])
    # log: 2..9 multiples of each power inside the limits
    s = scale_x_log10()
    s.train([1, 1000])
    assert len(s.get_minor_breaks()) == 3 * 8
    # user values (data space), callables, none
    s = scale_x_continuous(minor_breaks=[0.5, 2, 2.5, 99])
    s.train([0, 10])
    np.testing.assert_allclose(s.get_minor_breaks(), [0.5, 2.5])  # 2 is major, 99 outside
    s = scale_x_continuous(minor_breaks=breaks_width(0.5))
    s.train([0, 4])  # majors 0, 1, 2, 3, 4
    np.testing.assert_allclose(s.get_minor_breaks(), [0.5, 1.5, 2.5, 3.5])
    for off in (False, []):
        s = scale_x_continuous(minor_breaks=off)
        s.train([0, 10])
        assert len(s.get_minor_breaks()) == 0
    assert len(scale_x_continuous().get_minor_breaks()) == 0  # untrained


# --------------------------------------------------------------------------- coords
def _panel(xs=None, ys=None, th=None, x=(0, 10), y=(0, 5)):
    sx = xs or scale_x_continuous()
    sy = ys or scale_y_continuous()
    sx.train(list(x))
    sy.train(list(y))
    return coord_cartesian().build(sx, sy, th or theme_default(), {"x": "X", "y": "Y"})


def test_minor_grid_drawn():
    p = _panel()
    nmaj = len(p.panel_params.x_breaks) + len(p.panel_params.y_breaks)
    nmin = len(p.panel_params.x_minor_breaks) + len(p.panel_params.y_minor_breaks)
    assert nmin > 0
    # one GridLines mobject per axis (ggm-bam.2)
    assert len(p.grid_major) == 2 and len(p.grid_minor) == 2
    assert grid_line_count(p.grid_major) == nmaj and grid_line_count(p.grid_minor) == nmin
    assert grid_line_count(p.grid) == nmaj + nmin
    minor, major = p.grid_minor[0], p.grid_major[0]
    assert minor.get_stroke_width() < major.get_stroke_width()
    assert minor.get_stroke_opacity() < major.get_stroke_opacity()


def test_minor_grid_theme_switches():
    assert len(_panel(th=theme_classic()).grid) == 0
    off = theme_default().merge(theme(panel_grid=False))
    assert len(_panel(th=off).grid_minor) == 0
    no_minor = _panel(
        xs=scale_x_continuous(minor_breaks=False), ys=scale_y_continuous(minor_breaks=False)
    )
    assert len(no_minor.grid_minor) == 0 and len(no_minor.grid_major) > 0


def test_minor_grid_theme_fields():
    t = theme_default()
    assert len(_panel(th=t.merge(theme(panel_grid_minor=False))).grid_minor) == 0
    p = _panel(th=t.merge(theme(panel_grid_minor=False)))
    assert len(p.grid_major) > 0
    styled = _panel(
        th=t.merge(theme(grid_minor_stroke_width=0.7, grid_minor_opacity=0.3, grid_color="#ff0000"))
    )
    line = styled.grid_minor[0]
    assert line.get_stroke_width() == pytest.approx(0.7)
    assert line.get_stroke_opacity() == pytest.approx(0.3)
    assert line.get_stroke_color().to_hex().lower() == "#ff0000"
    # minor lines are independent of the per-axis major switches
    p = _panel(th=t.merge(theme(panel_grid_major_x=False)))
    assert grid_line_count(p.grid_minor) == len(p.panel_params.x_minor_breaks) + len(
        p.panel_params.y_minor_breaks
    )


@pytest.mark.parametrize("angle", [0, 45, 90, -45])
def test_top_right_rotated_labels_stay_outside(angle):
    th = theme_default().merge(theme(axis_text_x_angle=angle, axis_text_y_angle=angle))
    p = _panel(
        xs=scale_x_continuous(position="top"), ys=scale_y_continuous(position="right"), th=th
    )
    top_tick = max(t.get_top()[1] for t in p.x_ticks)
    right_tick = max(t.get_right()[0] for t in p.y_ticks)
    for lab, tick in zip(p.x_tick_labels, p.x_ticks):
        assert lab.get_bottom()[1] >= top_tick - 1e-6
        assert abs(lab.get_center()[0] - tick.get_center()[0]) < lab.width  # near its tick
    for lab, tick in zip(p.y_tick_labels, p.y_ticks):
        assert lab.get_left()[0] >= right_tick - 1e-6
        assert abs(lab.get_center()[1] - tick.get_center()[1]) < max(lab.height, 0.5)


def test_position_validation():
    assert scale_x_continuous().position == "bottom"
    assert scale_y_continuous().position == "left"
    assert scale_x_log10(position="top").position == "top"
    with pytest.raises(ValueError):
        scale_x_continuous(position="right")
    with pytest.raises(ValueError):
        scale_y_continuous(position="top")


def test_position_top_and_right_move_axes():
    base = _panel()
    moved = _panel(xs=scale_x_continuous(position="top"), ys=scale_y_continuous(position="right"))
    top_edge = moved.axes.c2p(0, 5)[1]
    right_edge = moved.axes.c2p(10, 0)[0]
    # tick labels above the panel / right of it; titles further out
    assert all(lab.get_bottom()[1] > top_edge for lab in moved.x_tick_labels)
    assert all(lab.get_left()[0] > right_edge for lab in moved.y_tick_labels)
    assert all(t.get_start()[1] >= top_edge - 1e-9 for t in moved.x_ticks)
    assert all(t.get_end()[1] > top_edge for t in moved.x_ticks)
    assert all(t.get_end()[0] > right_edge for t in moved.y_ticks)
    assert moved.x_title.get_bottom()[1] > max(lab.get_top()[1] for lab in moved.x_tick_labels)
    assert moved.y_title.get_left()[0] > max(lab.get_right()[0] for lab in moved.y_tick_labels)
    # default side unchanged
    bottom_edge = base.axes.c2p(0, 0)[1]
    assert all(lab.get_top()[1] < bottom_edge for lab in base.x_tick_labels)
    # geometry (c2p) is unaffected by moving the axis
    np.testing.assert_allclose(moved.transform([3], [2]), base.transform([3], [2]))


def test_label_math_renders_superscripts():
    sx = scale_x_log10(labels=label_math())
    p = _panel(xs=sx, x=(3, 2500))
    assert p.panel_params.x_labels == ("$10^{1}$", "$10^{2}$", "$10^{3}$")
    assert [lab.text for lab in p.x_tick_labels] == ["10¹", "10²", "10³"]


def test_full_build_with_helpers():
    df = pd.DataFrame({"x": [1, 10, 100, 1000], "y": [0.1, 0.25, 0.5, 0.9]})
    plot = (
        ggmanim.ggplot(df, ggmanim.aes("x", "y"))
        .geom_point()
        .scale_x_log10(labels=label_math(), position="top")
        .scale_y_continuous(labels=label_percent(), breaks=breaks_extended())
    )
    m = plot.build()
    panel = m.panel
    assert [lab.text for lab in panel.y_tick_labels][:2] == ["25%", "50%"]
    assert len(panel.grid_minor) > 0


def test_log_breaks_decreases_by_before_sub_breaks():
    # scales::log_breaks: 10^-0.2 .. 10^4.2 -> every power, not 1/5 sub-breaks
    np.testing.assert_allclose(log_breaks(10**-0.2, 10**4.2), [0.1, 1, 10, 100, 1e3, 1e4, 1e5])


def test_minor_grid_and_position_under_coord_flip():
    from ggmanim.coords import coord_flip

    sx, sy = scale_x_continuous(position="top"), scale_y_continuous(position="right")
    sx.train([0, 10])
    sy.train([0, 5])
    p = coord_flip().build(sx, sy, theme_default(), {"x": "X", "y": "Y"})
    pp = p.panel_params
    # panel_params stay in aesthetic orientation, minor breaks included
    np.testing.assert_allclose(pp.x_minor_breaks, sx.get_minor_breaks(pp.x_range))
    np.testing.assert_allclose(pp.y_minor_breaks, sy.get_minor_breaks(pp.y_range))
    assert grid_line_count(p.grid_minor) == len(pp.x_minor_breaks) + len(pp.y_minor_breaks)
    # x aesthetic is vertical: its "top" goes to the right; y's "right" to the top
    corner = p.axes.c2p(pp.y_range[1], pp.x_range[1])  # drawn axes are swapped
    right_edge, top_edge = corner[0], corner[1]
    assert all(lab.get_left()[0] > right_edge - 1e-6 for lab in p.y_tick_labels)
    assert all(lab.get_bottom()[1] > top_edge - 1e-6 for lab in p.x_tick_labels)
