"""Lines and intervals: stat_summary with pointranges / error bars / crossbars,
stairsteps, reference ablines, segments and curves with arrowheads.

Render a still of every scene::

    manim -ql -s -a examples/intervals.py

or one scene::

    manim -ql -s examples/intervals.py Summaries
"""

import numpy as np
import pandas as pd
from manim import Create, FadeIn, Scene

from ggmanim import (
    aes,
    arrow,
    geom_abline,
    geom_crossbar,
    geom_curve,
    geom_errorbar,
    geom_errorbarh,
    geom_linerange,
    geom_point,
    geom_pointrange,
    geom_segment,
    geom_step,
    ggplot,
    labs,
    stat_summary,
)


def measurements(seed: int = 3) -> pd.DataFrame:
    """Five treatments, 12 noisy measurements each."""
    rng = np.random.default_rng(seed)
    means = {"ctrl": 4.0, "A": 5.2, "B": 6.1, "C": 4.8, "D": 7.0}
    return pd.DataFrame(
        [
            {"treatment": t, "value": m + rng.normal(0, 0.9)}
            for t, m in means.items()
            for _ in range(12)
        ]
    )


def make_summaries():
    """Raw points, mean +/- 95% CI (error bars) and mean +/- SE (pointrange)."""
    df = measurements()
    return (
        ggplot(df, aes("treatment", "value"))
        .add(geom_point(alpha=0.35, size=0.04), name="points")
        .add(
            stat_summary(fun_data="mean_cl_normal", geom="errorbar", width=0.25, color="#E69F00"),
            name="ci",
        )
        .add(stat_summary(fun_data="mean_se", color="#0072B2"), name="se")
        .add(labs(title="stat_summary: mean ± SE and 95% CI", x="Treatment", y="Value"))
    )


def make_ranges():
    """The four interval geoms side by side, plus a horizontal error bar."""
    d = pd.DataFrame(
        {
            "geom": ["errorbar", "linerange", "pointrange", "crossbar"],
            "mid": [3.0, 4.0, 3.5, 5.0],
            "lo": [2.0, 2.5, 2.0, 4.0],
            "hi": [4.5, 5.5, 4.0, 6.5],
        }
    )
    m = aes(x="geom", y="mid", ymin="lo", ymax="hi")
    h = pd.DataFrame({"y": [1.2], "xmin": [1.5], "xmax": [3.5]})
    return (
        ggplot(d, m)
        .add(geom_errorbar(data=d.iloc[[0]], width=0.4), name="errorbar")
        .add(geom_linerange(data=d.iloc[[1]], linewidth=4), name="linerange")
        .add(geom_pointrange(data=d.iloc[[2]], color="#D55E00"), name="pointrange")
        .add(geom_crossbar(data=d.iloc[[3]], fill="#56B4E9", alpha=0.6, width=0.5), name="crossbar")
        .add(
            geom_errorbarh(
                aes(y="y", xmin="xmin", xmax="xmax"),
                h,
                height=0.4,
                inherit_aes=False,
                color="#009E73",
            ),
            name="errorbarh",
        )
        .add(labs(title="errorbar · linerange · pointrange · crossbar · errorbarh", x=None, y=None))
    )


def make_steps():
    """geom_step in its three directions, and a y = x reference abline."""
    x = np.arange(8.0)
    y = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 5.5, 7.5])
    df = pd.concat(
        [
            pd.DataFrame({"x": x, "y": y + off, "direction": d})
            for d, off in (("hv", 0.0), ("vh", 2.5), ("mid", 5.0))
        ],
        ignore_index=True,
    )
    p = ggplot(df, aes("x", "y", color="direction"))
    for d in ("hv", "vh", "mid"):
        p = p.add(geom_step(data=df[df.direction == d], direction=d), name=f"step_{d}")
    return (
        p.add(geom_point(size=0.05), name="points")
        .add(geom_abline(slope=1, intercept=0, linetype="dashed", linewidth=1.5), name="abline")
        .add(labs(title="geom_step(direction=...) and geom_abline", color="direction"))
    )


def make_arrows():
    """Segments and curves with open / closed arrowheads."""
    rng = np.random.default_rng(7)
    n = 6
    x, y = rng.uniform(0, 10, n), rng.uniform(0, 10, n)
    seg = pd.DataFrame({"x": x, "y": y, "xend": 10 - y, "yend": x})
    curve = pd.DataFrame({"x": [2.0, 8.0], "y": [0.5, 9.5], "xend": [8.0, 2.0], "yend": [0.5, 9.5]})
    return (
        ggplot(seg, aes("x", "y", xend="xend", yend="yend"))
        .add(geom_segment(arrow=arrow(), color="#0072B2"), name="segments")
        .add(
            geom_curve(
                data=curve, curvature=-0.3, arrow=arrow(type="closed", ends="both"), color="#D55E00"
            ),
            name="curves",
        )
        .add(geom_point(size=0.05), name="starts")
        .add(labs(title="geom_segment / geom_curve with arrow()"))
    )


class Summaries(Scene):
    def construct(self):
        plot = make_summaries().build()
        self.play(Create(plot.panel))
        self.play(FadeIn(plot.layers["points"], lag_ratio=0.02))
        self.play(Create(plot.layers["ci"]), FadeIn(plot.layers["se"]))
        self.add(plot)
        self.wait()


class Ranges(Scene):
    def construct(self):
        plot = make_ranges().build()
        self.play(FadeIn(plot))
        self.wait()


class Steps(Scene):
    def construct(self):
        plot = make_steps().build()
        self.play(Create(plot.panel))
        for d in ("hv", "vh", "mid"):
            self.play(Create(plot.layers[f"step_{d}"]), run_time=0.6)
        self.add(plot)
        self.wait()


class Arrows(Scene):
    def construct(self):
        plot = make_arrows().build()
        self.play(Create(plot.panel))
        self.play(Create(plot.layers["segments"], lag_ratio=0.2))
        self.add(plot)
        self.wait()
