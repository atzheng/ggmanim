"""``GGPlot.create()``: an ordered reveal with a natural animation per geom.

Render the animation (``-s`` gives only the last frame)::

    manim -ql examples/animate_create.py AnimateCreate
    manim -ql examples/animate_create.py AnimateFacets
"""

import numpy as np
import pandas as pd
from manim import Indicate, Scene

from ggmanim import aes, facet_wrap, geom_col, geom_point, geom_smooth, geom_text, ggplot, labs
from ggmanim.animate import attach_geoms


def sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["North", "South", "East", "West", "Central"],
            "change": [3.2, -1.5, 2.1, -0.6, 4.0],
        }
    )


def bar_plot():
    return (
        ggplot(sales(), aes("region", "change", key="region"))
        .add(geom_col(aes(fill="region"), show_legend=False), name="bars")
        .add(geom_text(aes(label="change"), nudge_y=0.4), name="labels")
        .add(labs(title="Change in sales", x=None, y="% change"))
    )


def scatter_plot():
    rng = np.random.default_rng(3)
    frames = []
    for site, slope in [("Site A", 0.8), ("Site B", 0.3), ("Site C", -0.4)]:
        x = rng.uniform(0, 10, 20)
        frames.append(pd.DataFrame({"x": x, "y": slope * x + rng.normal(0, 1, 20), "site": site}))
    df = pd.concat(frames, ignore_index=True)
    return (
        ggplot(df, aes("x", "y", color="site"))
        .add(geom_point(size=0.05), name="points")
        .add(geom_smooth(method="lm", se=False), name="fit")
        .add(facet_wrap("site"))
        .add(labs(title="Three sites"))
    )


class AnimateCreate(Scene):
    def construct(self):
        spec = bar_plot()
        plot = attach_geoms(spec.build(), spec)  # so bars grow from the zero line
        self.play(plot.create(lag_ratio=0.7))
        self.play(Indicate(plot.mark("bars", "East")))
        self.wait()


class AnimateFacets(Scene):
    def construct(self):
        spec = scatter_plot()
        plot = attach_geoms(spec.build(), spec)
        self.play(plot.create(panel_lag=0.25, run_time=4))
        self.play(plot.hide_layer("fit"))
        self.play(plot.reveal_layer("fit", panel_lag=0.3))
        self.wait()
