"""ggmanim.transition: data-keyed morphs between plots, states and reveal."""

import numpy as np
import pandas as pd
import pytest
from manim import FadeIn, FadeOut, Scene, Transform, Wait, tempconfig

from ggmanim import aes, geom_col, geom_line, geom_point, ggplot, labs, scale_x_log10
from ggmanim.mobjects import GGPlot
from ggmanim.transition import (
    StatesAnimation,
    Transition,
    _panel_box,
    animate_reveal,
    animate_states,
    shared_limits,
    transition,
)


def _frames():
    rng = np.random.default_rng(1)
    ids = [f"c{i}" for i in range(10)]
    a = pd.DataFrame({"id": ids, "x": rng.uniform(0, 10, 10), "y": rng.uniform(0, 5, 10)})
    # 8 of 10 keys shared, 2 leave, 2 arrive
    b = pd.DataFrame(
        {"id": ids[2:] + ["n1", "n2"], "x": rng.uniform(0, 20, 10), "y": rng.uniform(0, 5, 10)}
    )
    return a, b


def _scatter(df):
    return ggplot(df, aes("x", "y")).add(geom_point(), name="pts")


def _leaves(anim):
    subs = getattr(anim, "animations", None)
    if not subs:
        return [anim]
    return [leaf for a in subs for leaf in _leaves(a)]


def _play(*fns):
    with tempconfig({"dry_run": True, "quality": "low_quality", "disable_caching": True}):

        class S(Scene):
            def construct(self):
                for fn in fns:
                    out = fn(self)
                    if out is not None:
                        self.play(out)

        scene = S()
        scene.render()
    return scene


# -- matching ----------------------------------------------------------------------


def test_shared_keys_transform_rest_fades():
    a, b = _frames()
    t = transition(_scatter(a), _scatter(b), key="id")
    assert isinstance(t, Transition)
    assert t.layer_stats["pts"] == {"matched": 8, "exited": 2, "entered": 2}
    kinds = [type(x) for x in t.layer_animations["pts"]]
    assert kinds.count(Transform) == 8
    assert kinds.count(FadeOut) == 2
    assert kinds.count(FadeIn) == 2


def test_matched_marks_target_the_same_key():
    a, b = _frames()
    t = transition(_scatter(a), _scatter(b), key="id")
    marks_b = t.plot.layers["pts"].marks
    keys_b = t.plot.layers["pts"].keys
    working = t.working.layers["pts"]
    pairs = {
        id(an.mobject): an.target_mobject
        for an in t.layer_animations["pts"]
        if type(an) is Transform
    }
    for m, k in zip(working.marks, working.keys):
        if id(m) in pairs:
            assert keys_b[marks_b.index(pairs[id(m)])] == k


def test_without_key_matches_row_index():
    a, b = _frames()
    t = transition(_scatter(a), _scatter(b))
    assert t.layer_stats["pts"] == {"matched": 10, "exited": 0, "entered": 0}


def test_key_already_mapped_on_built_plots():
    a, b = _frames()
    ga = ggplot(a, aes("x", "y", key="id")).add(geom_point(), name="pts").build()
    gb = ggplot(b, aes("x", "y", key="id")).add(geom_point(), name="pts").build()
    t = transition(ga, gb)
    assert t.plot is gb
    assert t.layer_stats["pts"]["matched"] == 8


def test_unmatched_layers_fade_and_by_position():
    a, b = _frames()
    pa = _scatter(a).add(geom_line(), name="line")
    pb = ggplot(b, aes("x", "y")).add(geom_point(), name="dots")
    t = transition(pa, pb)
    assert t.layer_stats == {}
    t2 = transition(pa, pb, match_layers="by_position")
    assert set(t2.layer_stats) == {"dots"}
    with pytest.raises(ValueError, match="match_layers"):
        transition(pa, pb, match_layers="nope")


def test_every_part_is_animated():
    """Nothing of either plot pops: every leaf is covered by some animation."""
    a, b = _frames()
    t = transition(_scatter(a).add(labs(title="A")), _scatter(b).add(labs(title="B")), key="id")
    covered_a, covered_b = set(), set()
    for an in _leaves(t):
        if isinstance(an, FadeIn):
            covered_b.update(id(m) for m in an.mobject.get_family())
        elif isinstance(an, Transform) and not isinstance(an, FadeOut):
            covered_a.update(id(m) for m in an.mobject.get_family())
            covered_b.update(id(m) for m in an.target_mobject.get_family())
        else:
            covered_a.update(id(m) for m in an.mobject.get_family())
    assert all(id(m) in covered_a for m in t.working.family_members_with_points())
    assert all(id(m) in covered_b for m in t.plot.family_members_with_points())


# -- axes and layout ----------------------------------------------------------------


def _axis_df(hi):
    return pd.DataFrame({"x": [0.0, hi], "y": [0.0, 1.0]})


def test_axis_morph_keeps_shared_tick_labels():
    t = transition(_scatter(_axis_df(10)), _scatter(_axis_df(20)))
    la = t.working.panel.panel_params.x_labels
    lb = t.plot.panel.panel_params.x_labels
    shared = set(la) & set(lb)
    assert shared  # e.g. "0", "5", "10"
    texts_b = list(t.plot.panel.x_tick_labels)
    moved = [
        lb[texts_b.index(an.target_mobject)]
        for an in _leaves(t)
        if type(an) is Transform and an.target_mobject in texts_b
    ]
    assert sorted(moved) == sorted(shared)
    # the others fade, sliding along the axis
    labels_a = list(t.working.panel.x_tick_labels)
    outgoing = [an for an in _leaves(t) if isinstance(an, FadeOut) and an.mobject in labels_a]
    assert len(outgoing) == len(set(la) - shared)


def test_panels_are_aligned():
    a, b = _frames()
    t = transition(
        _scatter(a),
        ggplot(b, aes("x", "y", colour="id")).add(geom_point(), name="pts"),  # with a legend
    )
    lo_a, hi_a = _panel_box(t.plot_a)
    lo_b, hi_b = _panel_box(t.plot)
    np.testing.assert_allclose(lo_a, lo_b, atol=1e-6)
    np.testing.assert_allclose(hi_a, hi_b, atol=1e-6)


def test_limits_union_and_explicit():
    a, b = _frames()
    t = transition(_scatter(a), _scatter(b), limits="union")
    assert t.plot_a.panel.panel_params.x_range == pytest.approx(t.plot.panel.panel_params.x_range)
    lim = shared_limits(_scatter(a), _scatter(b))
    assert lim["x"][0] == pytest.approx(min(a.x.min(), b.x.min()))
    assert lim["x"][1] == pytest.approx(max(a.x.max(), b.x.max()))
    t2 = transition(_scatter(a), _scatter(b), limits={"x": (0, 30)})
    assert t2.plot.scales["x"].get_limits() == pytest.approx((0, 30))
    assert t2.plot_a.scales["x"].get_limits() == pytest.approx((0, 30))
    t3 = transition(_scatter(a), _scatter(b), limits="a")
    assert t3.plot.scales["x"].get_limits() == pytest.approx(t3.plot_a.scales["x"].get_limits())
    with pytest.raises(ValueError, match="limits"):
        transition(_scatter(a), _scatter(b), limits="both")


def test_limits_keep_log_scale():
    df = pd.DataFrame({"x": [10.0, 1000.0], "y": [1.0, 2.0]})
    df2 = pd.DataFrame({"x": [100.0, 10000.0], "y": [1.0, 2.0]})
    t = transition(
        _scatter(df).add(scale_x_log10()), _scatter(df2).add(scale_x_log10()), limits="union"
    )
    assert t.plot.scales["x"].get_limits() == pytest.approx((1.0, 4.0))


def test_discrete_bars_resort():
    def bars(order, vals):
        d = pd.DataFrame({"k": order, "v": vals})
        d["k"] = pd.Categorical(d.k, categories=order)
        return ggplot(d, aes("k", "v")).add(geom_col(), name="bars")

    t = transition(bars(["a", "b", "c"], [1, 2, 3]), bars(["c", "a", "b"], [1, 5, 6]), key="k")
    assert t.layer_stats["bars"]["matched"] == 3
    # every discrete tick label is shared (matched by text, not position)
    texts_b = list(t.plot.panel.x_tick_labels)
    assert sum(type(an) is Transform and an.target_mobject in texts_b for an in _leaves(t)) == 3


# -- in a scene ----------------------------------------------------------------------


def test_scene_ends_with_second_plot_and_chains():
    a, b = _frames()
    box = {}

    def first(scene):
        t = transition(_scatter(a), _scatter(b), key="id", run_time=0.2)
        box["t"] = t
        scene.add(t.plot_a)
        return t

    def second(scene):
        assert scene.mobjects == [box["t"].plot]
        t2 = transition(box["t"].plot, _scatter(a), key="id", run_time=0.2)
        box["t2"] = t2
        return t2

    scene = _play(first, second)
    assert scene.mobjects == [box["t2"].plot]
    assert isinstance(box["t2"].plot, GGPlot)


def test_animate_states():
    a, b = _frames()
    df = pd.concat([a.assign(yr=2000), b.assign(yr=2010), a.assign(yr=2020)])
    spec = _scatter(df)
    anim = animate_states(spec, "yr", key="id", title="Year {state}", state_length=0.1)
    assert isinstance(anim, StatesAnimation)
    assert anim.states == [2000, 2010, 2020]
    assert len(anim.transitions) == 2 and len(anim.plots) == 3
    assert [type(x) for x in anim.animations] == [Wait, Transition, Wait, Transition, Wait]
    assert anim.transitions[0].layer_stats["pts"] == {"matched": 8, "exited": 2, "entered": 2}
    title = anim.plots[1].title
    assert "2010" in (getattr(title, "original_text", None) or title.text)
    # shared limits ("union") across states
    ranges = {p.panel.panel_params.x_range for p in anim.plots}
    assert len(ranges) == 1
    wrapped = animate_states(spec, "yr", key="id", wrap=True, state_length=0)
    assert len(wrapped.transitions) == 3
    assert wrapped.plots[-1] is not wrapped.plots[0]

    scene = _play(
        lambda s: animate_states(spec, "yr", key="id", transition_length=0.2, state_length=0.1)
    )
    assert len(scene.mobjects) == 1 and isinstance(scene.mobjects[0], GGPlot)


def test_animate_states_errors():
    a, _ = _frames()
    with pytest.raises(ValueError, match="column"):
        animate_states(_scatter(a), "nope")


def test_animate_reveal():
    x = np.linspace(0, 10, 30)
    df = pd.concat([pd.DataFrame({"x": x, "y": np.sin(x) + i, "s": s}) for i, s in enumerate("ab")])
    spec = ggplot(df, aes("x", "y", colour="s")).add(geom_line(), name="lines")
    anim = animate_reveal(spec, steps=5, run_time=1)
    assert len(anim.plots) == 5
    assert anim.states[-1] == pytest.approx(10.0)
    for t in anim.transitions:
        assert t.layer_stats["lines"] == {"matched": 2, "exited": 0, "entered": 0}
    assert len({p.panel.panel_params.x_range for p in anim.plots}) == 1
    # lines grow
    widths = [p.layers["lines"].marks[0].width for p in anim.plots]
    assert widths == sorted(widths) and widths[0] < widths[-1]
    with pytest.raises(ValueError, match="along"):
        animate_reveal(ggplot(df).add(geom_line(aes("x", "y"))))
    scene = _play(lambda s: animate_reveal(spec, steps=3, run_time=0.3))
    assert len(scene.mobjects) == 1
