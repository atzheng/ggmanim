"""Counting stats: a stacked histogram by group, frequency polygons, density
histogram, and dodged bar counts of a string column.

Render a still of the last frame of every scene::

    manim -ql -s -a examples/histograms.py
"""

import numpy as np
import pandas as pd
from manim import ORIGIN, Create, FadeIn, Scene, VGroup, config

from ggmanim import (
    aes,
    after_stat,
    geom_bar,
    geom_freqpoly,
    geom_histogram,
    geom_text,
    ggplot,
    labs,
    position_dodge,
    theme,
)


def heights(seed: int = 7) -> pd.DataFrame:
    """Two overlapping normal samples."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "height": np.concatenate([rng.normal(164, 7, 300), rng.normal(177, 8, 300)]),
            "sex": np.repeat(["F", "M"], 300),
        }
    )


def diamonds_like(seed: int = 2) -> pd.DataFrame:
    """A small categorical table (diamond cut by colour group)."""
    rng = np.random.default_rng(seed)
    cuts = ["Fair", "Good", "VG", "Prem", "Ideal"]
    cut = rng.choice(cuts, size=400, p=[0.05, 0.12, 0.23, 0.26, 0.34])
    df = pd.DataFrame({"cut": pd.Categorical(cut, cuts), "clarity": rng.choice(["I1", "VS"], 400)})
    return df


def make_plots():
    df = heights()
    cuts = diamonds_like()
    return [
        ggplot(df, aes("height", fill="sex")).add(
            geom_histogram(binwidth=2.5, color="#222222"),
            labs(title="geom_histogram(binwidth=2.5)", fill="Sex"),
        ),
        ggplot(df, aes("height", color="sex")).add(
            geom_freqpoly(binwidth=2.5), labs(title="geom_freqpoly()", color="Sex")
        ),
        ggplot(df, aes("height")).add(
            geom_histogram(aes(y=after_stat("density")), bins=25, fill="#4C78A8"),
            labs(title='after_stat("density")'),
        ),
        ggplot(cuts, aes("cut", fill="clarity")).add(
            geom_bar(position="dodge"),
            geom_text(
                aes(label=after_stat("count"), group="clarity"),
                stat="count",
                position=position_dodge(width=0.9),
                vjust=-0.3,
                size=2.5,
                show_legend=False,
            ),
            labs(title='geom_bar(position="dodge")', x=None, fill="Clarity"),
            theme(legend_position="bottom"),
        ),
    ]


def fit_frame(group: VGroup, margin: float = 0.3) -> VGroup:
    w, h = config.frame_width - 2 * margin, config.frame_height - 2 * margin
    if group.width > w or group.height > h:
        group.scale(min(w / group.width, h / group.height))
    return group.move_to(ORIGIN)


class Histograms(Scene):
    def construct(self):
        plots = [p.add(theme(panel_width=3.6, panel_height=2.0)).build() for p in make_plots()]
        grid = fit_frame(VGroup(*plots).arrange_in_grid(rows=2, cols=2, buff=0.4))
        self.play(*(Create(p.panel) for p in plots))
        self.play(*(FadeIn(p.layers[0], lag_ratio=0.02) for p in plots))
        self.add(grid)
        self.wait()
