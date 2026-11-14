"""Blank a mapped or default fill/colour with ``None`` / ``"NA"`` (ggplot2's ``NA``).

Left: ``geom_density(aes(fill="g", color="g"), fill=None)`` keeps only the
coloured outlines although ``fill`` is mapped. Right: a boxplot with
``fill="NA"`` over a mapped fill, and filled points (``shape=21``) whose outline
is removed with ``color="none"``.

Render a still of the last frame::

    manim -ql -s examples/blank_fill.py BlankFill
"""

import numpy as np
import pandas as pd
from manim import RIGHT, Create, FadeIn, Scene, VGroup

from ggmanim import aes, geom_boxplot, geom_density, geom_jitter, ggplot, labs, theme

SIZE = theme(panel_width=4.6, panel_height=3.6)


def samples(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = [
        pd.DataFrame({"g": g, "v": rng.normal(mu, sd, 120)})
        for g, mu, sd in [("a", 0.0, 1.0), ("b", 1.5, 0.7), ("c", -1.0, 1.4)]
    ]
    return pd.concat(parts, ignore_index=True)


def make_density():
    return (
        ggplot(samples(), aes("v", fill="g", color="g"))
        .add(geom_density(fill=None, linewidth=4))
        .add(labs(title="geom_density(fill=None)", x="value"), SIZE)
    )


def make_boxplot():
    return (
        ggplot(samples(), aes("g", "v", fill="g"))
        .add(geom_boxplot(fill="NA", outliers=False))
        .add(geom_jitter(shape=21, size=0.05, color="none", alpha=0.7, width=0.2, seed=1))
        .add(labs(title='fill="NA" box, color="none" points', x="group", y="value"), SIZE)
    )


class BlankFill(Scene):
    def construct(self):
        left, right = make_density().build(), make_boxplot().build()
        VGroup(left, right).arrange(RIGHT, buff=0.4).scale_to_fit_width(13.4)
        self.play(Create(left.panel), Create(right.panel))
        self.play(FadeIn(left.layers[0]), FadeIn(right.layers[0]), FadeIn(right.layers[1]))
        self.play(FadeIn(left), FadeIn(right))  # titles, axes labels and legends
        self.wait()
