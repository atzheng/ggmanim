"""ggm-f4q: transition() shares discrete non-position scales; discrete axis
labels do not overlap while cross-fading."""

import pandas as pd
from manim import FadeIn, FadeOut

from ggmanim import aes, geom_col, geom_point, ggplot, scale_colour_discrete
from ggmanim.transition import transition


def _leaves(anim):
    subs = getattr(anim, "animations", None)
    if not subs:
        return [anim]
    return [leaf for a in subs for leaf in _leaves(a)]


def _pts(levels):
    df = pd.DataFrame({"x": range(len(levels)), "y": range(len(levels)), "g": levels})
    return ggplot(df, aes("x", "y", colour="g")).add(geom_point())


def _limits(plot):
    return list(plot.scales["color"].get_limits())


def test_same_level_same_colour():
    t = transition(_pts(["u", "v"]), _pts(["v", "w"]))
    assert _limits(t.plot_a) == _limits(t.plot) == ["u", "v", "w"]
    sa, sb = t.plot_a.scales["color"], t.plot.scales["color"]
    assert sa.map(["v"]) == sb.map(["v"])


def test_user_limits_kept():
    a = _pts(["u", "v"]).add(scale_colour_discrete(limits=["u", "v"]))
    t = transition(a, _pts(["v", "w"]))
    assert _limits(t.plot_a) == ["u", "v"]


def test_opt_out():
    t = transition(_pts(["u", "v"]), _pts(["v", "w"]), share_scales=False)
    assert _limits(t.plot) == ["v", "w"]


def test_discrete_labels_do_not_overlap():
    def bars(cats):
        df = pd.DataFrame({"c": cats, "n": [1] * len(cats)})
        return ggplot(df, aes("c", "n")).add(geom_col())

    t = transition(bars(["a", "b"]), bars(["c", "d"]))
    outs = [x for x in _leaves(t) if isinstance(x, FadeOut)]
    ins = [x for x in _leaves(t) if isinstance(x, FadeIn)]
    assert outs and ins
    assert any(o.rate_func(0.5) == 1.0 and o.rate_func(0.1) < 1.0 for o in outs)
    assert any(i.rate_func(0.5) == 0.0 and i.rate_func(0.9) > 0.0 for i in ins)
