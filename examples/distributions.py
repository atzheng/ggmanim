"""Distributions: boxplots, violins, density curves and rugs.

Render a still of every scene::

    manim -ql -s -a examples/distributions.py

or one scene::

    manim -ql -s examples/distributions.py Boxplots
"""

import numpy as np
import pandas as pd
from manim import DOWN, Create, FadeIn, GrowFromEdge, Scene

from ggmanim import (
    aes,
    geom_boxplot,
    geom_density,
    geom_jitter,
    geom_point,
    geom_rug,
    geom_violin,
    ggplot,
    labs,
    position_jitterdodge,
)


def reaction_times(seed: int = 11) -> pd.DataFrame:
    """Reaction times for four tasks and two groups, skewed, with a few outliers."""
    rng = np.random.default_rng(seed)
    base = {"read": 420.0, "count": 510.0, "rotate": 640.0, "recall": 560.0}
    rows = []
    for task, mu in base.items():
        for group, shift in (("young", 0.0), ("older", 70.0)):
            vals = mu + shift + rng.gamma(4.0, 25.0, size=30) - 100
            vals[rng.integers(0, 30, size=2)] += rng.choice([-1, 1]) * rng.uniform(220, 320)
            rows += [{"task": task, "group": group, "ms": v} for v in vals]
    df = pd.DataFrame(rows)
    df["task"] = pd.Categorical(df["task"], categories=list(base))
    return df


def make_boxplots():
    """Grouped (dodged) boxplots with outliers and the raw points jittered on top."""
    df = reaction_times()
    return (
        ggplot(df, aes("task", "ms", fill="group"))
        .add(geom_boxplot(alpha=0.6, outlier_shape="circle"), name="boxes")
        .add(
            geom_point(
                aes(color="group"),
                position=position_jitterdodge(jitter_width=0.15, dodge_width=0.75, seed=1),
                size=0.025,
                alpha=0.6,
            ),
            name="points",
        )
        .add(labs(title="geom_boxplot + jittered points", x="Task", y="Reaction time (ms)"))
    )


def make_violins():
    """Horizontal violins (``y`` discrete), dodged by group, with quartile lines."""
    df = reaction_times()
    return (
        ggplot(df, aes("ms", "task", fill="group"))
        .add(
            geom_violin(alpha=0.5, scale="width", draw_quantiles=[0.25, 0.5, 0.75]),
            name="violins",
        )
        .add(labs(title="Horizontal violins with quartiles", x="Reaction time (ms)", y=None))
    )


def make_densities():
    """Overlapping kernel densities by group, with a rug of the observations."""
    df = reaction_times()
    return (
        ggplot(df, aes("ms", fill="group", color="group"))
        .add(geom_density(alpha=0.35, adjust=0.8), name="densities")
        .add(geom_rug(alpha=0.6, linewidth=1.0), name="rug")
        .add(labs(title="geom_density(adjust=0.8) + geom_rug()", x="Reaction time (ms)"))
    )


def make_gallery():
    """The gallery scene: violin + narrow boxplot + jittered points per task."""
    df = reaction_times()
    return (
        ggplot(df, aes("task", "ms"))
        .add(geom_violin(aes(fill="task"), alpha=0.35, show_legend=False), name="violins")
        .add(geom_boxplot(width=0.15, outliers=False, linewidth=1.5), name="boxes")
        .add(geom_jitter(width=0.08, size=0.02, alpha=0.5), name="points")
        .add(labs(title="violin + boxplot + jitter", x=None, y="Reaction time (ms)"))
    )


class Boxplots(Scene):
    def construct(self):
        plot = make_boxplots().build()
        self.play(Create(plot.panel))
        self.play(*[GrowFromEdge(m, DOWN) for m in plot.layers["boxes"].marks], run_time=1.2)
        self.play(FadeIn(plot.layers["points"], lag_ratio=0.01))
        self.add(plot)
        self.wait()


class Violins(Scene):
    def construct(self):
        plot = make_violins().build()
        self.play(Create(plot.panel))
        self.play(FadeIn(plot.layers["violins"], lag_ratio=0.2))
        self.add(plot)
        self.wait()


class Densities(Scene):
    def construct(self):
        plot = make_densities().build()
        self.play(Create(plot.panel))
        self.play(Create(plot.layers["densities"]), run_time=1.5)
        self.play(FadeIn(plot.layers["rug"], lag_ratio=0.01))
        self.add(plot)
        self.wait()


class Gallery(Scene):
    def construct(self):
        plot = make_gallery().build()
        self.play(FadeIn(plot))
        self.wait()
