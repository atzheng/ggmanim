"""stat_summary_2d / stat_summary_hex, weighted density_2d and an ordered fill
(viridis by default), with a long title above a legend taller than its panel.

Render stills of the last frame::

    manim -ql -s examples/summary_2d.py Summary2d
"""

import numpy as np
import pandas as pd
from manim import ORIGIN, UP, Create, FadeIn, Scene, VGroup, config

from ggmanim import (
    aes,
    geom_col,
    geom_density_2d_filled,
    ggplot,
    labs,
    stat_summary_2d,
    stat_summary_hex,
    theme,
)


def surface(n: int = 1500, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(-3, 3, n), rng.uniform(-3, 3, n)
    z = np.sin(x) * np.cos(y) + rng.normal(0, 0.15, n)
    return pd.DataFrame({"x": x, "y": y, "z": z, "w": np.exp(-(x**2 + y**2) / 4)})


def make_summary_2d():
    return (
        ggplot(surface(), aes("x", "y", z="z"))
        .add(stat_summary_2d(bins=12))
        .add(labs(title="stat_summary_2d: mean of z", fill="mean z"))
    )


def make_summary_hex():
    return (
        ggplot(surface(), aes("x", "y", z="z"))
        .add(stat_summary_hex(bins=12, fun="median"))
        .add(labs(title="stat_summary_hex: median of z", fill="median z"))
    )


def make_weighted_density():
    return (
        ggplot(surface(), aes("x", "y", weight="w"))
        .add(geom_density_2d_filled(bins=8))
        .add(labs(title="density_2d weighted towards the centre", fill="level"))
    )


def make_ordered():
    sizes = ["XS", "S", "M", "L", "XL"]
    df = pd.DataFrame(
        {"size": pd.Categorical(sizes, sizes, ordered=True), "n": [3.0, 7.0, 12.0, 8.0, 4.0]}
    )
    return (
        ggplot(df, aes("size", "n", fill="size"))
        .add(geom_col())
        .add(labs(title="Ordered factor: viridis by default"))
    )


def fit_frame(group: VGroup, margin: float = 0.3) -> VGroup:
    w, h = config.frame_width - 2 * margin, config.frame_height - 2 * margin
    if group.width > w or group.height > h:
        group.scale(min(w / group.width, h / group.height))
    return group.move_to(ORIGIN)


class Summary2d(Scene):
    def construct(self):
        small = theme(panel_width=3.2, panel_height=2.0)
        makers = (make_summary_2d, make_summary_hex, make_weighted_density, make_ordered)
        plots = [m().add(small).build() for m in makers]
        grid = VGroup(*plots).arrange_in_grid(rows=2, cols=2, buff=(0.6, 0.5))
        fit_frame(grid)
        self.play(*(Create(p.panel) for p in plots))
        self.play(*(FadeIn(p.layers[0], shift=UP * 0.1) for p in plots))
        self.add(grid)
        self.wait()
