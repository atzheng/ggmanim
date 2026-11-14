"""Position adjustments: stacked, filled and dodged bars; jittered and
jitter-dodged points.

Render a still of the last frame of every scene::

    manim -ql -s -a examples/positions.py
"""

import numpy as np
import pandas as pd
from manim import ORIGIN, Create, FadeIn, Scene, VGroup, config

from ggmanim import (
    aes,
    geom_col,
    geom_jitter,
    geom_point,
    ggplot,
    labs,
    position_dodge,
    position_jitterdodge,
    theme,
)


def sales() -> pd.DataFrame:
    """Quarterly sales per region, with one negative (returns) quarter."""
    return pd.DataFrame(
        {
            "quarter": np.repeat(["Q1", "Q2", "Q3", "Q4"], 3),
            "region": ["North", "South", "West"] * 4,
            "sales": [4, 3, 2, 5, 2, 3, 6, 4, -1.5, 3, 5, 4],
        }
    )


def measurements(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i, dose in enumerate(["low", "mid", "high"]):
        for j, sex in enumerate(["F", "M"]):
            y = 2 + i + 0.6 * j + rng.normal(0, 0.5, 15)
            rows.append(pd.DataFrame({"dose": dose, "sex": sex, "response": y}))
    df = pd.concat(rows, ignore_index=True)
    df["dose"] = pd.Categorical(df["dose"], ["low", "mid", "high"])
    return df


def make_bar_plots():
    base = ggplot(sales(), aes("quarter", "sales", fill="region"))
    # a sparser table (Q1 South and Q4 South/West missing) shows the dodge widths
    sparse = sales().drop(index=[1, 10, 11])
    dodged = ggplot(sparse, aes("quarter", "sales", fill="region"))
    return [
        base.add(geom_col(), labs(title="stack (default)", x=None, fill="Region")),
        base.add(geom_col(position="fill"), labs(title='"fill"', x=None, y="share")),
        dodged.add(geom_col(position="dodge"), labs(title='"dodge"', x=None)),
        dodged.add(
            geom_col(position=position_dodge(preserve="single")),
            labs(title='dodge(preserve="single")', x=None),
        ),
    ]


def make_point_plots():
    df = measurements()
    return [
        ggplot(df, aes("dose", "response")).add(
            geom_jitter(width=0.25, height=0, alpha=0.8, seed=1), labs(title="geom_jitter()")
        ),
        ggplot(df, aes("dose", "response", color="sex")).add(
            geom_point(position=position_jitterdodge(dodge_width=0.6, seed=1), alpha=0.8),
            labs(title="position_jitterdodge()", color="Sex"),
        ),
    ]


def fit_frame(group: VGroup, margin: float = 0.3) -> VGroup:
    w, h = config.frame_width - 2 * margin, config.frame_height - 2 * margin
    if group.width > w or group.height > h:
        group.scale(min(w / group.width, h / group.height))
    return group.move_to(ORIGIN)


def build_small(plots, width: float, height: float, **kw) -> list:
    return [p.add(theme(panel_width=width, panel_height=height, **kw)).build() for p in plots]


class StackFillDodge(Scene):
    def construct(self):
        plots = build_small(make_bar_plots(), 3.2, 1.9, legend_position="none")
        plots[0] = make_bar_plots()[0].add(theme(panel_width=3.2, panel_height=1.9)).build()
        grid = fit_frame(VGroup(*plots).arrange_in_grid(rows=2, cols=2, buff=0.35))
        self.play(*(Create(p.panel) for p in plots))
        self.play(*(FadeIn(p.layers[0], lag_ratio=0.05) for p in plots))
        self.add(grid)
        self.wait()


class Jitter(Scene):
    def construct(self):
        plots = build_small(make_point_plots(), 4.2, 3.6)
        row = fit_frame(VGroup(*plots).arrange(buff=0.4))
        self.play(*(Create(p.panel) for p in plots))
        self.play(*(FadeIn(p.layers[0], lag_ratio=0.02) for p in plots))
        self.add(row)
        self.wait()
