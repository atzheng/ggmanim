"""Smoothers and distribution stats: loess / lm / a user model with fullrange,
an ECDF per group, a normal Q-Q plot and overplotted counts.

Render a still of the last frame::

    manim -ql -s examples/smoothers.py Smoothers
"""

import numpy as np
import pandas as pd
from manim import ORIGIN, Create, FadeIn, Scene, VGroup, config

from ggmanim import (
    aes,
    geom_count,
    geom_point,
    geom_qq,
    geom_qq_line,
    geom_smooth,
    ggplot,
    labs,
    stat_ecdf,
    theme,
    xlim,
)


def kernel_smoother(x, y, bandwidth=0.8):
    """A user model for ``geom_smooth(method=...)``: Nadaraya-Watson regression.

    Returns a predictor ``f(xgrid) -> fit`` (return ``(fit, se)`` for a band)."""

    def predict(xgrid):
        w = np.exp(-0.5 * ((xgrid[:, None] - x[None, :]) / bandwidth) ** 2)
        return (w @ y) / w.sum(axis=1)

    return predict


def wiggly(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 10, 80))
    return pd.DataFrame({"x": x, "y": np.sin(x) + 0.25 * x + rng.normal(0, 0.35, len(x))})


def make_smooths():
    return (
        ggplot(wiggly(), aes("x", "y"))
        .add(geom_point(alpha=0.6, size=0.04), name="points")
        .add(geom_smooth(method="lm", color="#999999", se=False, fullrange=True), name="lm")
        .add(geom_smooth(method="loess", span=0.4), name="loess")
        .add(
            geom_smooth(method=kernel_smoother, se=False, color="#E69F00", linetype="dashed"),
            name="kernel",
        )
        .add(xlim(-1, 11), labs(title="lm (fullrange), loess, user model"))
    )


def samples(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(0, 1, 60), rng.exponential(1.0, 60) - 1]),
            "dist": ["normal"] * 60 + ["exponential"] * 60,
        }
    )


def make_ecdf():
    return (
        ggplot(samples(), aes("value", color="dist"))
        .add(stat_ecdf(), name="ecdf")
        .add(labs(title="stat_ecdf", y="F(value)", color=None))
    )


def make_qq():
    df = samples().query("dist == 'normal'")
    return (
        ggplot(df, aes(sample="value"))
        .add(geom_qq_line(color="#D55E00"), name="line")
        .add(geom_qq(size=0.05), name="points")
        .add(labs(title="geom_qq + geom_qq_line", x="theoretical", y="sample"))
    )


def make_count():
    rng = np.random.default_rng(11)
    df = pd.DataFrame({"x": rng.poisson(2.5, 150), "y": rng.poisson(1.5, 150)})
    return (
        ggplot(df, aes("x", "y"))
        .add(geom_count(alpha=0.8), name="counts")
        .add(labs(title="geom_count", size="n"))
    )


def build_all(width: float = 3.6, height: float = 2.2) -> list:
    plots = [make_smooths(), make_ecdf(), make_qq(), make_count()]
    return [p.add(theme(panel_width=width, panel_height=height)).build() for p in plots]


class Smoothers(Scene):
    def construct(self):
        grid = VGroup(*build_all()).arrange_in_grid(rows=2, cols=2, buff=0.3)
        w, h = config.frame_width - 0.6, config.frame_height - 0.6
        if grid.width > w or grid.height > h:
            grid.scale(min(w / grid.width, h / grid.height))
        grid.move_to(ORIGIN)
        self.play(FadeIn(grid))
        self.play(*(Create(p.layers[k]) for p, k in zip(grid, ["loess", "ecdf", "line", "counts"])))
        self.wait()
