"""Secondary axes: sec_axis / dup_axis (ggm-aji.3)."""

import numpy as np
import pandas as pd
import pytest

from ggmanim import (
    aes,
    coord_flip,
    coord_trans,
    derive,
    dup_axis,
    facet_wrap,
    ggplot,
    labs,
    scale_x_continuous,
    scale_x_log10,
    scale_y_continuous,
    sec_axis,
    theme,
)
from ggmanim.scales.continuous import Y_AESTHETICS, ScaleContinuousPosition
from ggmanim.scales.sec_axis import AxisSecondary


def f_of_c(c):
    return c * 1.8 + 32


def _temps():
    return pd.DataFrame({"t": np.arange(10.0), "c": np.linspace(-5, 35, 10)})


def _celsius_plot(**sec):
    return (
        ggplot(_temps(), aes("t", "c"))
        .geom_line()
        .add(scale_y_continuous(name="°C", sec_axis=sec_axis(f_of_c, name="°F", **sec)))
    )


# ---------------------------------------------------------------- spec / scale
def test_constructors_and_validation():
    s = sec_axis(f_of_c, name="F")
    assert isinstance(s, AxisSecondary) and s.name == "F" and s.transform is f_of_c
    assert sec_axis(trans=f_of_c).transform is f_of_c
    with pytest.raises(TypeError):
        sec_axis(f_of_c, trans=f_of_c)
    with pytest.raises(TypeError):
        sec_axis("not callable")
    with pytest.raises(ValueError):
        sec_axis(f_of_c, breaks=[1, 2], labels=["a"])
    d = dup_axis()
    assert d.name is derive() and d.breaks is derive() and d.labels is derive()
    assert scale_x_continuous(sec_axis=d).sec_axis is d
    assert scale_x_log10(sec_axis=d).sec_axis is d
    assert scale_y_continuous().sec_axis is None
    with pytest.raises(TypeError):
        scale_y_continuous(sec_axis=lambda x: x)


def test_break_info_exact_celsius_fahrenheit():
    sc = ScaleContinuousPosition(Y_AESTHETICS)
    sc.train([-5, 35])
    rng = sc.get_expanded_limits()
    pos, labs_ = sec_axis(f_of_c).break_info(sc, rng)
    assert labs_ == ("20", "40", "60", "80")
    for p, lab in zip(pos, labs_):
        assert abs(f_of_c(p) - float(lab)) < 1e-6
        assert rng[0] <= p <= rng[1]


def test_break_info_decreasing_transform_and_user_breaks():
    sc = ScaleContinuousPosition(Y_AESTHETICS)
    sc.train([0, 10])
    pos, labs_ = sec_axis(lambda x: -2 * x, breaks=[-20, -10, 0]).break_info(
        sc, sc.get_expanded_limits()
    )
    assert labs_ == ("0", "-10", "-20")  # ordered along the primary axis
    np.testing.assert_allclose(pos, [0, 5, 10], atol=1e-9)


def test_break_info_non_monotone_warns():
    sc = ScaleContinuousPosition(Y_AESTHETICS)
    sc.train([-1, 1])
    with pytest.warns(UserWarning, match="monotone"):
        assert sec_axis(lambda x: x**2).break_info(sc, (-1.0, 1.0)) == ((), ())


def test_discrete_primary_errors():
    class Fake:
        is_discrete = True

    with pytest.raises(TypeError, match="continuous"):
        sec_axis(f_of_c).break_info(Fake(), (0.0, 1.0))


# ---------------------------------------------------------------- drawing
def test_secondary_axis_drawn_on_right_at_converted_positions():
    g = _celsius_plot().build()
    pan = g.panel
    assert pan.y_sec_side == "right" and pan.x_sec_side is None
    assert len(pan.x_sec_ticks) == 0 and pan.x_sec_title is None
    assert pan.y_sec_labels == ("20", "40", "60", "80")
    for p, lab in zip(pan.y_sec_breaks, pan.y_sec_labels):
        assert abs(f_of_c(p) - float(lab)) < 1e-6
    # ticks on the right edge, at the screen heights of the primary positions
    right = pan.axes.c2p(pan.panel_params.x_range[1], 0)[0]
    assert len(pan.y_sec_ticks) == len(pan.y_sec_tick_labels) == 4
    for tick, p in zip(pan.y_sec_ticks, pan.y_sec_breaks):
        start = tick.get_start()
        assert abs(start[0] - right) < 1e-6
        assert abs(start[1] - pan.transform(0, p)[0, 1]) < 1e-6
        assert tick.get_end()[0] > start[0]  # drawn outward
    for lab in pan.y_sec_tick_labels:
        assert lab.get_left()[0] > right
    assert pan.y_sec_title.text == "°F"
    assert pan.y_sec_title.get_left()[0] > max(m.get_right()[0] for m in pan.y_sec_tick_labels)
    assert pan.y_sec_line is not None


def test_primary_on_right_puts_secondary_left():
    p = (
        ggplot(_temps(), aes("t", "c"))
        .geom_line()
        .add(scale_y_continuous(position="right", sec_axis=sec_axis(f_of_c)))
    )
    pan = p.build().panel
    assert pan.y_sec_side == "left"
    left = pan.axes.c2p(pan.panel_params.x_range[0], 0)[0]
    assert all(t.get_end()[0] < left for t in pan.y_sec_ticks)
    assert pan.y_sec_title is None  # sec_axis() has no title by default


def test_dup_axis_mirrors_primary():
    df = pd.DataFrame({"x": [0.5, 3000.0], "y": [1.0, 2.0]})
    p = ggplot(df, aes("x", "y")).geom_point().add(scale_x_log10(name="mass", sec_axis=dup_axis()))
    pan = p.build().panel
    pp = pan.panel_params
    assert pan.x_sec_side == "top"
    np.testing.assert_allclose(pan.x_sec_breaks, pp.x_breaks, atol=1e-6)
    assert pan.x_sec_labels == pp.x_labels
    assert pan.x_sec_title.text == "mass"
    top = pan.axes.c2p(0, pp.y_range[1])[1]
    assert all(t.get_start()[1] == pytest.approx(top) for t in pan.x_sec_ticks)


def test_dup_axis_title_from_labs_and_derived_labels():
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    p = (
        ggplot(df, aes("x", "y"))
        .geom_point()
        .add(
            scale_y_continuous(labels=lambda b: [f"{v:.0%}" for v in b], sec_axis=dup_axis()),
            labs(y="share"),
        )
    )
    pan = p.build().panel
    assert pan.y_sec_title.text == "share"
    assert pan.y_sec_labels == pan.panel_params.y_labels
    assert pan.y_sec_labels[-1] == "100%"


def test_primary_formatter_used_for_secondary_labels():
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    p = (
        ggplot(df, aes("x", "y"))
        .geom_point()
        .add(
            scale_y_continuous(
                labels=lambda b: [f"<{v:g}>" for v in b], sec_axis=sec_axis(lambda v: v * 10)
            )
        )
    )
    pan = p.build().panel
    assert pan.y_sec_labels and all(s.startswith("<") for s in pan.y_sec_labels)
    for p_, s in zip(pan.y_sec_breaks, pan.y_sec_labels):
        assert abs(p_ * 10 - float(s.strip("<>"))) < 1e-6


def test_theme_switches_hide_secondary_parts():
    pan = _celsius_plot().add(theme(axis_ticks_y=False, axis_title_y=False)).build().panel
    assert len(pan.y_sec_ticks) == 0 and pan.y_sec_title is None
    assert len(pan.y_sec_tick_labels) == 4


def test_coord_flip_secondary_axis_on_top():
    pan = _celsius_plot().add(coord_flip()).build().panel
    # the y aesthetic (°C) is horizontal: its secondary axis goes on top
    assert pan.x_sec_side == "top" and pan.y_sec_side is None
    assert pan.x_sec_labels == ("20", "40", "60", "80")
    top = pan.axes.c2p(0, pan.axes.y_range[1])[1]
    for tick, p in zip(pan.x_sec_ticks, pan.x_sec_breaks):
        assert tick.get_start()[1] == pytest.approx(top)
        assert tick.get_start()[0] == pytest.approx(pan.transform(0, p)[0, 0])
        assert abs(f_of_c(p) - float(pan.x_sec_labels[list(pan.x_sec_breaks).index(p)])) < 1e-6
    assert pan.x_sec_title.text == "°F"
    assert pan.x_sec_title.get_bottom()[1] > top


def test_coord_trans_secondary_ticks_match_primary():
    df = pd.DataFrame({"x": [1.0, 1000.0], "y": [0.0, 1.0]})
    p = (
        ggplot(df, aes("x", "y"))
        .geom_point()
        .add(scale_x_continuous(breaks=[1, 10, 100, 1000], sec_axis=dup_axis()))
        .add(coord_trans(x="log10"))
    )
    pan = p.build().panel
    assert pan.x_sec_labels == ("1", "10", "100", "1000")
    np.testing.assert_allclose(
        [t.get_start()[0] for t in pan.x_sec_ticks], [t.get_start()[0] for t in pan.x_ticks]
    )


# ---------------------------------------------------------------- facets
def _faceted():
    df = pd.concat(
        [_temps().assign(city=c, c=_temps()["c"] + i) for i, c in enumerate("ABCDE")],
        ignore_index=True,
    )
    return (
        ggplot(df, aes("t", "c"))
        .geom_line()
        .add(
            facet_wrap("city", ncol=3),
            scale_y_continuous(name="°C", sec_axis=sec_axis(f_of_c, name="°F")),
            scale_x_continuous(sec_axis=dup_axis()),
        )
    )


def test_facets_secondary_on_outer_panels_only():
    g = _faceted().build()
    panels = g.panels  # {(row, col): panel}; A B C / D E
    has_right = {rc for rc, p in panels.items() if len(p.y_sec_tick_labels)}
    has_top = {rc for rc, p in panels.items() if len(p.x_sec_tick_labels)}
    has_left = {rc for rc, p in panels.items() if len(p.y_tick_labels)}
    has_bottom = {rc for rc, p in panels.items() if len(p.x_tick_labels)}
    assert has_right == {(1, 3), (2, 2)}
    assert has_top == {(1, 1), (1, 2), (1, 3)}
    assert has_left == {(1, 1), (2, 1)}
    assert has_bottom == {(2, 1), (2, 2), (1, 3)}
    # titles once for the grid, panels have none
    assert all(p.y_sec_title is None and p.x_sec_title is None for p in panels.values())
    grid = g.panel_grid
    assert grid.y_sec_title.text == "°F"
    assert grid.x_sec_title.text == "t"
    assert grid.y_sec_title.get_left()[0] > max(
        m.get_right()[0] for p in panels.values() for m in p.y_sec_tick_labels
    )
    # top labels sit above the strips (strip_placement="inside")
    strip_top = max(s.get_top()[1] for s in grid.strips)
    assert min(m.get_bottom()[1] for m in panels[(1, 1)].x_sec_tick_labels) > strip_top - 1e-6


def test_facets_with_coord_flip():
    g = _faceted().add(coord_flip()).build()
    panels = g.panels
    # under flip the °C (y aesthetic) axis is horizontal: secondary on the top row
    has_top = {rc for rc, p in panels.items() if len(p.x_sec_tick_labels)}
    has_right = {rc for rc, p in panels.items() if len(p.y_sec_tick_labels)}
    assert has_top == {(1, 1), (1, 2), (1, 3)}
    assert has_right == {(1, 3), (2, 2)}
    pan = panels[(1, 1)]
    assert pan.x_sec_labels[:4] == ("20", "40", "60", "80")
    for p, lab in zip(pan.x_sec_breaks, pan.x_sec_labels):
        assert abs(f_of_c(p) - float(lab)) < 1e-6
    assert g.panel_grid.x_sec_title.text == "°F"
    assert g.panel_grid.y_sec_title.text == "t"


def test_no_secondary_attributes_are_empty():
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    pan = ggplot(df, aes("x", "y")).geom_point().build().panel
    assert pan.x_sec_side is None and pan.y_sec_side is None
    assert len(pan.x_sec_ticks) == 0 and pan.y_sec_breaks == ()
    assert pan.secondary_parts == {}
