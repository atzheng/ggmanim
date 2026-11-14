"""ggm-g15: transition follow-ups (top-level exports, date/discrete limits,
stable group ids in animate_states, ticks fading at the panel edge)."""

import numpy as np
import pandas as pd
import pytest
from manim import FadeIn, FadeOut

import ggmanim
from ggmanim import aes, geom_col, geom_line, geom_point, ggplot, scale_x_date
from ggmanim.scales.date import ScaleDatePosition
from ggmanim.scales.discrete_position import ScaleDiscretePosition
from ggmanim.transition import (
    EDGE_FADE,
    Transition,
    _AxisSlide,
    shared_limits,
)
from ggmanim.transition import (
    animate_states as _animate_states,
)
from ggmanim.transition import (
    transition as _transition,
)


def _leaves(anim):
    subs = getattr(anim, "animations", None)
    if not subs:
        return [anim]
    return [leaf for a in subs for leaf in _leaves(a)]


def _scatter(df):
    return ggplot(df, aes("x", "y")).add(geom_point(), name="pts")


# -- exports ---------------------------------------------------------------------


def test_top_level_exports():
    assert ggmanim.transition is _transition
    assert ggmanim.animate_states is _animate_states
    assert callable(ggmanim.animate_reveal)
    for name in ("transition", "animate_states", "animate_reveal"):
        assert name in ggmanim.__all__


def test_ggplot_transition_to():
    a = pd.DataFrame({"id": list("abc"), "x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    b = a.assign(x=[3.0, 1.0, 2.0])
    built = ggplot(a, aes("x", "y", key="id")).add(geom_point(), name="pts").build()
    t = built.transition_to(_scatter(b), key="id", run_time=0.5)
    assert isinstance(t, Transition)
    assert t.plot_a is built
    assert t.layer_stats["pts"]["matched"] == 3
    assert t.run_time == pytest.approx(0.5)


# -- discrete limits ----------------------------------------------------------------


def _bars(levels, vals):
    d = pd.DataFrame({"k": levels, "v": vals})
    return ggplot(d, aes("k", "v")).add(geom_col(), name="bars")


def test_shared_limits_discrete_union():
    lim = shared_limits(_bars(["a", "c"], [1, 2]), _bars(["b", "c", "d"], [1, 2, 3]))
    assert lim["x"] == ["a", "b", "c", "d"]  # sorted inputs: sorted union
    # category order is kept (first appearance) when an input is not sorted
    d = pd.DataFrame({"k": pd.Categorical(["z", "a"], categories=["z", "a"]), "v": [1, 2]})
    lim2 = shared_limits(ggplot(d, aes("k", "v")).add(geom_col()), _bars(["b"], [1]))
    assert lim2["x"] == ["z", "a", "b"]


def test_transition_discrete_union_keeps_positions():
    t = _transition(_bars(["a", "b"], [1, 2]), _bars(["b", "c"], [3, 4]), key="k", limits="union")
    for p in (t.plot_a, t.plot):
        assert isinstance(p.scales["x"], ScaleDiscretePosition)
        assert p.scales["x"].get_limits() == ["a", "b", "c"]
    assert t.plot_a.panel.panel_params.x_range == pytest.approx(t.plot.panel.panel_params.x_range)
    # the shared bar "b" stays at the same x
    xa = t.plot_a.layers["bars"].marks[1].get_center()[0]
    xb = t.plot.layers["bars"].marks[0].get_center()[0]
    assert xa == pytest.approx(xb, abs=1e-6)
    assert t.layer_stats["bars"] == {"matched": 1, "exited": 1, "entered": 1}


def test_transition_discrete_explicit_levels():
    t = _transition(
        _bars(["a", "b"], [1, 2]), _bars(["b", "c"], [3, 4]), limits={"x": ["c", "b", "a"]}
    )
    assert t.plot.scales["x"].get_limits() == ["c", "b", "a"]
    assert t.plot_a.scales["x"].get_limits() == ["c", "b", "a"]


def test_animate_states_discrete_levels_shared():
    df = pd.DataFrame(
        {"k": ["a", "b", "b", "c", "a", "c"], "v": [1, 2, 3, 4, 5, 6], "s": [1, 1, 2, 2, 3, 3]}
    )
    anim = _animate_states(_bars(df.k, df.v)._replace(data=df), "s", state_length=0)
    for p in anim.plots:
        assert p.scales["x"].get_limits() == ["a", "b", "c"]


# -- date limits --------------------------------------------------------------------


def _dates(start, n):
    d = pd.date_range(start, periods=n, freq="MS")
    return pd.DataFrame({"x": d, "y": np.arange(n, dtype=float)})


def test_shared_limits_dates():
    a, b = _scatter(_dates("2020-01-01", 6)), _scatter(_dates("2020-04-01", 9))
    lim = shared_limits(a, b)
    lo, hi = lim["x"]
    assert lo == pytest.approx(pd.Timestamp("2020-01-01").value / 86400e9)
    assert hi == pytest.approx(pd.Timestamp("2020-12-01").value / 86400e9)


def test_transition_dates_union():
    t = _transition(
        _scatter(_dates("2020-01-01", 6)), _scatter(_dates("2020-04-01", 9)), limits="union"
    )
    for p in (t.plot_a, t.plot):
        assert isinstance(p.scales["x"], ScaleDatePosition)
    assert t.plot_a.panel.panel_params.x_range == pytest.approx(t.plot.panel.panel_params.x_range)
    # tick labels are still dates
    assert any("2020" in s or "Jan" in s or "-" in s for s in t.plot.panel.panel_params.x_labels)


def test_transition_dates_explicit_and_user_scale():
    spec_a = _scatter(_dates("2020-01-01", 6)).add(scale_x_date(date_labels="%b"))
    t = _transition(
        spec_a, _scatter(_dates("2020-04-01", 3)), limits={"x": ("2019-12-01", "2021-01-01")}
    )
    days = pd.Timestamp("2019-12-01").value / 86400e9, pd.Timestamp("2021-01-01").value / 86400e9
    assert t.plot_a.scales["x"].get_limits() == pytest.approx(days)
    assert t.plot.scales["x"].get_limits() == pytest.approx(days)
    assert isinstance(t.plot.scales["x"], ScaleDatePosition)
    # spec_a's own scale (with its labels) was kept
    assert all(len(s) == 3 for s in t.plot_a.panel.panel_params.x_labels)


def test_animate_states_dates():
    df = pd.concat([_dates("2020-01-01", 4).assign(s=1), _dates("2020-06-01", 4).assign(s=2)])
    anim = _animate_states(_scatter(df), "s", state_length=0)
    r = {p.panel.panel_params.x_range for p in anim.plots}
    assert len(r) == 1
    assert all(isinstance(p.scales["x"], ScaleDatePosition) for p in anim.plots)


# -- stable group ids ----------------------------------------------------------------


def _series():
    x = np.arange(5, dtype=float)
    parts = []
    for s, series in ((1, "bc"), (2, "abc"), (3, "ac")):
        for name in series:
            parts.append(pd.DataFrame({"x": x, "y": x * (ord(name) - 96) + s, "g": name, "s": s}))
    return pd.concat(parts, ignore_index=True)


def test_animate_states_stable_group_ids():
    spec = ggplot(_series(), aes("x", "y", colour="g")).add(geom_line(), name="lines")
    anim = _animate_states(spec, "s", state_length=0)
    keys = [p.layers["lines"].keys for p in anim.plots]
    # a=1, b=2, c=3 in every state, whichever series are present
    assert keys == [[2, 3], [1, 2, 3], [1, 3]]
    assert anim.transitions[0].layer_stats["lines"] == {"matched": 2, "exited": 0, "entered": 1}
    assert anim.transitions[1].layer_stats["lines"] == {"matched": 2, "exited": 1, "entered": 0}
    # the "b" line of state 1 morphs into the "b" line of state 2 (same colour)
    t = anim.transitions[0]
    working = t.working.layers["lines"].marks
    target = {
        id(an.mobject): an.target_mobject
        for an in t.layer_animations["lines"]
        if type(an) not in (FadeIn, FadeOut)
    }
    b_target = target[id(working[0])]
    assert b_target is t.plot.layers["lines"].marks[1]
    assert working[0].get_stroke_color() == b_target.get_stroke_color()
    # colours (and the legend) are those of the full data in every state
    assert [len(p.legend.family_members_with_points()) for p in anim.plots] == [
        len(anim.plots[1].legend.family_members_with_points())
    ] * 3
    # no extra legend or axis title from the hidden group column
    p = anim.plots[1]
    assert len(p.legends) == 1
    assert ".ggm" not in str(p.panel.panel_params.x_labels)


def test_stable_groups_respect_user_group_and_layer_data():
    df = _series()
    spec = ggplot(df, aes("x", "y", group="g")).add(geom_line(), name="lines")
    anim = _animate_states(spec, "s", state_length=0)
    # user ``group`` is left alone: ids re-coded per state (1..k)
    assert anim.plots[0].layers["lines"].keys == [1, 2]
    spec2 = ggplot().add(geom_line(aes("x", "y", colour="g"), data=df), name="lines")
    anim2 = _animate_states(spec2, "s", state_length=0)
    assert [p.layers["lines"].keys for p in anim2.plots] == [[2, 3], [1, 2, 3], [1, 3]]


# -- ticks at the panel edge -----------------------------------------------------------


def _slides(t):
    return [an for an in _leaves(t) if isinstance(an, _AxisSlide)]


def test_unmatched_ticks_do_not_pile_up_at_the_edge():
    small = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 5.0]})
    big = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 100.0]})
    t = _transition(_scatter(small), _scatter(big))
    slides = _slides(t)
    assert slides
    assert any(isinstance(s, FadeIn) for s in slides) and any(
        isinstance(s, FadeOut) for s in slides
    )
    labels = set(map(id, t.working.panel.y_tick_labels)) | set(map(id, t.plot.panel.y_tick_labels))
    ylab = [s for s in slides if id(s.mobject) in labels]
    assert ylab
    for alpha in np.linspace(0, 1, 41):
        at_edge = 0
        for s in ylab:
            pc, f = s.place(alpha)
            assert s.lo - 1e-9 <= pc <= s.hi + 1e-9
            p = s.p0 + (s.p1 - s.p0) * alpha
            if p > s.hi + EDGE_FADE or p < s.lo - EDGE_FADE:
                assert f == 0.0
            if f > 0.05 and min(abs(pc - s.lo), abs(pc - s.hi)) < 1e-9:
                at_edge += 1
        assert at_edge <= 1


def test_axis_slide_moves_and_fades_the_mobject():
    small = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 5.0]})
    big = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 100.0]})
    t = _transition(_scatter(small), _scatter(big))
    s = next(
        an
        for an in _slides(t)
        if isinstance(an, FadeIn) and an.mobject in list(t.plot.panel.y_tick_labels)
    )
    home_y = s.mobject.get_center()[1]
    s.begin()
    s.interpolate(0.0)
    # starts invisible, clamped inside the panel
    glyph = s.mobject.family_members_with_points()[0]
    assert glyph.get_fill_opacity() == pytest.approx(0.0)
    assert s.lo - 0.5 <= s.mobject.get_center()[1] <= s.hi + 0.5
    s.interpolate(1.0)
    assert s.mobject.get_center()[1] == pytest.approx(home_y, abs=1e-6)
    assert glyph.get_fill_opacity() == pytest.approx(1.0)
