"""Horizontal counting stats: ``geom_bar(aes(y=...))`` and
``geom_histogram(aes(y=...))`` count along y and draw bars from the y axis
(stacked and dodged along x), as in ggplot2.

Render a still of the last frame::

    manim -ql -s examples/horizontal_bars.py HorizontalBars
"""

import numpy as np
import pandas as pd
from manim import ORIGIN, Create, FadeIn, Scene, VGroup, config

from ggmanim import aes, geom_bar, geom_freqpoly, geom_histogram, ggplot, labs, theme


def sample(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cuts = ["Fair", "Good", "VG", "Prem", "Ideal"]
    n = 400
    return pd.DataFrame(
        {
            "cut": pd.Categorical(rng.choice(cuts, n, p=[0.05, 0.12, 0.23, 0.26, 0.34]), cuts),
            "clarity": rng.choice(["I1", "VS"], n),
            "height": np.concatenate([rng.normal(164, 7, n // 2), rng.normal(177, 8, n // 2)]),
            "sex": np.repeat(["F", "M"], n // 2),
        }
    )


def make_plots():
    df = sample()
    return [
        ggplot(df, aes(y="cut", fill="clarity")).add(
            geom_bar(), labs(title="geom_bar(aes(y=cut))", y=None, fill="Clarity")
        ),
        ggplot(df).add(
            geom_bar(aes(y="cut", fill="clarity"), position="dodge"),
            labs(title='position="dodge"', y=None, fill="Clarity"),
        ),
        ggplot(df, aes(y="height", fill="sex")).add(
            geom_histogram(binwidth=2.5, color="#222222"),
            labs(title="geom_histogram(aes(y=height))", fill="Sex"),
        ),
        ggplot(df, aes(y="height", color="sex")).add(
            geom_freqpoly(binwidth=2.5), labs(title="geom_freqpoly(aes(y=height))", color="Sex")
        ),
    ]


def fit_frame(group: VGroup, margin: float = 0.3) -> VGroup:
    w, h = config.frame_width - 2 * margin, config.frame_height - 2 * margin
    if group.width > w or group.height > h:
        group.scale(min(w / group.width, h / group.height))
    return group.move_to(ORIGIN)


class HorizontalBars(Scene):
    def construct(self):
        plots = [p.add(theme(panel_width=3.6, panel_height=2.0)).build() for p in make_plots()]
        grid = fit_frame(VGroup(*plots).arrange_in_grid(rows=2, cols=2, buff=0.4))
        self.play(*(Create(p.panel) for p in plots))
        self.play(*(FadeIn(p.layers[0], lag_ratio=0.02) for p in plots))
        self.add(grid)
        self.wait()
