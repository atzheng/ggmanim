"""``facet_wrap``: seven panels wrapped into a 3 x 3 grid.

Render a still of the last frame of each scene::

    manim -ql -s examples/facet_wrap.py FacetWrap
    manim -ql -s examples/facet_wrap.py FacetWrapFreeY
    manim -ql -s examples/facet_wrap.py FacetWrapVertical
    manim -ql -s examples/facet_wrap.py FacetWrapStripLeft
"""

import numpy as np
import pandas as pd
from manim import Create, FadeIn, Scene, VGroup, Write

from ggmanim import aes, facet_wrap, geom_point, geom_smooth, ggplot, label_both, labs

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def sales(seed: int = 3) -> pd.DataFrame:
    """Synthetic hourly sales for seven weekdays; weekends sell a lot more."""
    rng = np.random.default_rng(seed)
    frames = []
    for i, day in enumerate(DAYS):
        hour = rng.uniform(8, 20, size=14)
        scale = 3.0 if day in ("Sat", "Sun") else 1.0 + 0.1 * i
        units = scale * (5 + 0.8 * (hour - 8)) + rng.normal(0, 1.5 * scale, size=14)
        frames.append(pd.DataFrame({"hour": hour, "units": units, "day": day}))
    df = pd.concat(frames, ignore_index=True)
    df["day"] = pd.Categorical(df["day"], categories=DAYS)
    return df


def make_plot(**wrap):
    return (
        ggplot(sales(), aes("hour", "units"))
        .add(geom_point(alpha=0.8, size=0.05), name="points")
        .add(geom_smooth(method="lm", se=False), name="fit")
        .add(facet_wrap("day", **wrap), labs(x="Hour", y="Units sold"))
    )


class FacetWrap(Scene):
    """7 panels -> 3 x 3; x tick labels on the bottom panel of each column
    (Fri/Sat in row 2 sit above the gap), y on the left column."""

    wrap: dict = {}

    def construct(self):
        plot = make_plot(**self.wrap).build()
        self.play(Create(VGroup(*plot.panels.values())))
        self.play(Write(plot.strips))
        self.play(FadeIn(plot.layers["points"], lag_ratio=0.01))
        self.play(Create(plot.layers["fit"]))
        self.add(plot)  # axis titles and anything else not animated above
        self.wait()


class FacetWrapFreeY(FacetWrap):
    """Free y scales: every panel draws its own y axis; ``label_both`` strips."""

    wrap: dict = {"scales": "free_y", "labeller": label_both}


class FacetWrapVertical(FacetWrap):
    """``dir="v"``: panels fill columns first, here in 2 rows."""

    wrap: dict = {"dir": "v", "nrow": 2}


class FacetWrapStripLeft(FacetWrap):
    """Strips on the left of each panel (between the y axis and the panel)."""

    wrap: dict = {"strip_position": "left"}
