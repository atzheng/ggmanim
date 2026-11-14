"""2-D stats: bin2d, hex, density contours (lines and filled bands), contours of a z grid.

Render stills of the last frame::

    manim -ql -s examples/two_d.py TwoD
    manim -ql -s examples/two_d.py DensityFilled
"""

import numpy as np
import pandas as pd
from manim import ORIGIN, UP, Create, FadeIn, Scene, VGroup, config

from ggmanim import (
    aes,
    after_stat,
    geom_bin_2d,
    geom_contour,
    geom_contour_filled,
    geom_density_2d,
    geom_density_2d_filled,
    geom_hex,
    geom_point,
    ggplot,
    labs,
    scale_fill_viridis_c,
    scale_fill_viridis_d,
    theme,
)


def cloud(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.multivariate_normal([0, 0], [[1.0, 0.6], [0.6, 1.0]], size=n // 2)
    b = rng.multivariate_normal([2.5, -1.0], [[0.4, 0.0], [0.0, 0.3]], size=n // 2)
    xy = np.vstack([a, b])
    return pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]})


def field(n: int = 60) -> pd.DataFrame:
    x = np.linspace(-3, 3, n)
    xx, yy = np.meshgrid(x, x)
    z = np.exp(-((xx - 1) ** 2 + yy**2)) + 0.8 * np.exp(-((xx + 1.2) ** 2 + (yy - 1) ** 2) / 0.5)
    return pd.DataFrame({"x": xx.ravel(), "y": yy.ravel(), "z": z.ravel()})


def make_bin2d():
    return (
        ggplot(cloud(), aes("x", "y"))
        .add(geom_bin_2d(bins=20), scale_fill_viridis_c())
        .add(labs(title="geom_bin_2d", fill="count"))
    )


def make_hex():
    return (
        ggplot(cloud(), aes("x", "y"))
        .add(geom_hex(bins=15), scale_fill_viridis_c(option="magma"))
        .add(labs(title="geom_hex", fill="count"))
    )


def make_density():
    return (
        ggplot(cloud(), aes("x", "y"))
        .add(geom_point(size=0.02, alpha=0.4), geom_density_2d())
        .add(labs(title="geom_density_2d"))
    )


def make_density_filled():
    return (
        ggplot(cloud(), aes("x", "y"))
        .add(geom_density_2d_filled(), name="bands")
        .add(scale_fill_viridis_d(), labs(title="density_2d_filled", fill="density"))
    )


def make_contour():
    return (
        ggplot(field(), aes("x", "y", z="z"))
        .add(geom_contour_filled(bins=8), name="bands")
        .add(geom_contour(bins=8, color="white", linewidth=1.0))
        .add(scale_fill_viridis_d(option="cividis"), labs(title="contour(_filled)", fill="z"))
    )


def make_contour_level():
    return (
        ggplot(field(), aes("x", "y", z="z"))
        .add(geom_contour(aes(color=after_stat("level")), binwidth=0.1))
        .add(labs(title="contour, color=level", color="level"))
    )


def fit_frame(group: VGroup, margin: float = 0.3) -> VGroup:
    w, h = config.frame_width - 2 * margin, config.frame_height - 2 * margin
    if group.width > w or group.height > h:
        group.scale(min(w / group.width, h / group.height))
    return group.move_to(ORIGIN)


class TwoD(Scene):
    def construct(self):
        small = theme(panel_width=3.2, panel_height=2.4)
        makers = (
            make_bin2d,
            make_hex,
            make_density,
            make_density_filled,
            make_contour,
            make_contour_level,
        )
        plots = [m().add(small).build() for m in makers]
        grid = VGroup(*plots).arrange_in_grid(rows=2, cols=3, buff=(0.5, 0.5))
        fit_frame(grid)
        self.play(*(Create(p.panel) for p in plots))
        self.play(*(FadeIn(p.layers[0], shift=UP * 0.1) for p in plots))
        self.add(grid)
        self.wait()


class DensityFilled(Scene):
    def construct(self):
        plot = make_density_filled().build()
        self.play(Create(plot.panel))
        self.play(FadeIn(plot.layers["bands"], lag_ratio=0.1))
        self.add(plot)
        self.wait()
