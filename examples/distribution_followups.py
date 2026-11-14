"""Distribution follow-ups: horizontal positions, faceted boxplots, bandwidth rules
and horizontal densities from the plot mapping.

Render a still of every scene::

    manim -ql -s -a examples/distribution_followups.py

or one scene::

    manim -ql -s examples/distribution_followups.py FacetedHorizontalBoxplots
"""

import numpy as np
import pandas as pd
from manim import Create, FadeIn, Scene

from ggmanim import aes, facet_wrap, geom_boxplot, geom_density, geom_violin, ggplot, labs


def reaction_times(seed: int = 11) -> pd.DataFrame:
    """Reaction times for three tasks, two groups and two sessions."""
    rng = np.random.default_rng(seed)
    base = {"read": 420.0, "count": 510.0, "rotate": 640.0}
    rows = []
    for task, mu in base.items():
        for group, shift in (("young", 0.0), ("older", 70.0)):
            for session, drift in (("day 1", 0.0), ("day 2", -40.0)):
                vals = mu + shift + drift + rng.gamma(4.0, 25.0, size=25) - 100
                vals[rng.integers(0, 25)] += rng.uniform(200, 300)
                rows += [{"task": task, "group": group, "session": session, "ms": v} for v in vals]
    df = pd.DataFrame(rows)
    df["task"] = pd.Categorical(df["task"], categories=list(base))
    return df


def bimodal(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    v = np.concatenate([rng.normal(-1.5, 0.5, 120), rng.normal(1.2, 0.9, 180)])
    return pd.DataFrame({"value": v})


def make_faceted_boxplots():
    """Horizontal boxplots (``y`` discrete) dodged along y by group, one panel per
    session; ``position_dodge2`` flips itself for the horizontal layer."""
    return (
        ggplot(reaction_times(), aes("ms", "task", fill="group"))
        .add(geom_boxplot(alpha=0.6), name="boxes")
        .add(facet_wrap("session"))
        .add(labs(title="Horizontal boxplots + facet_wrap", x="Reaction time (ms)", y=None))
    )


def make_faceted_violins():
    """Vertical violins dodged by group in each session panel."""
    return (
        ggplot(reaction_times(), aes("task", "ms", fill="group"))
        .add(geom_violin(alpha=0.5, draw_quantiles=[0.5]), name="violins")
        .add(facet_wrap("session"))
        .add(labs(title="Violins + facet_wrap", x=None, y="Reaction time (ms)"))
    )


def make_bandwidths():
    """Horizontal densities (only ``y`` in the *plot* mapping) with three
    bandwidth rules: R's default ``nrd0``, Sheather-Jones and unbiased CV."""
    return (
        ggplot(bimodal(), aes(y="value"))
        .add(geom_density(bw="nrd0", color="#888888", linewidth=3), name="nrd0")
        .add(geom_density(bw="SJ", color="#1f77b4", linewidth=3), name="sj")
        .add(geom_density(bw="ucv", color="#d62728", linewidth=3), name="ucv")
        .add(
            labs(
                title='bw = "nrd0" (grey), "SJ" (blue), "ucv" (red)',
                x="density",
                y="value",
            )
        )
    )


class FacetedHorizontalBoxplots(Scene):
    def construct(self):
        plot = make_faceted_boxplots().build()
        self.play(FadeIn(plot))
        self.wait()


class FacetedViolins(Scene):
    def construct(self):
        plot = make_faceted_violins().build()
        self.play(FadeIn(plot))
        self.wait()


class Bandwidths(Scene):
    def construct(self):
        plot = make_bandwidths().build()
        self.play(Create(plot.panel))
        for name in ("nrd0", "sj", "ucv"):
            self.play(Create(plot.layers[name]), run_time=1.0)
        self.add(plot)
        self.wait()
