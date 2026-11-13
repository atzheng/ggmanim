"""Scatter + linear smooth per group, colour legend, log10 x axis.

Render a still of the last frame::

    manim -ql -s examples/basic.py Basic

or the animation::

    manim -ql examples/basic.py Basic
"""

import numpy as np
import pandas as pd
from manim import Create, FadeIn, Scene, Write

from ggmanim import aes, geom_point, geom_smooth, ggplot, labs, scale_x_log10


def dose_response(seed: int = 1) -> pd.DataFrame:
    """Synthetic dose-response data: three compounds, response linear in log10(dose)."""
    rng = np.random.default_rng(seed)
    frames = []
    for compound, slope, intercept in [("A", 1.6, 1.0), ("B", 1.0, 2.0), ("C", 0.4, 2.5)]:
        dose = 10 ** rng.uniform(0, 3, size=25)
        response = intercept + slope * np.log10(dose) + rng.normal(0, 0.35, size=25)
        frames.append(pd.DataFrame({"dose": dose, "response": response, "compound": compound}))
    return pd.concat(frames, ignore_index=True)


def make_plot():
    return (
        ggplot(dose_response(), aes("dose", "response", color="compound"))
        .add(geom_point(alpha=0.8), name="points")
        .add(geom_smooth(method="lm"), name="fit")
        .add(scale_x_log10(), labs(x="Dose (mg)", y="Response", color="Compound"))
    )


class Basic(Scene):
    def construct(self):
        plot = make_plot().build()
        self.play(Create(plot.panel))
        self.play(FadeIn(plot.layers["points"], lag_ratio=0.02))
        self.play(Create(plot.layers["fit"]))
        self.play(Write(plot.legend))
        self.wait()
