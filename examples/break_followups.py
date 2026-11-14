"""Break density follow-ups (ggm-udx).

* ``coord_trans`` axes thin their automatic breaks in small panels, like
  cartesian ones.
* Under ``coord_flip`` a facet's ``scales="free_x"`` frees the *horizontal* axis
  (the y aesthetic), as in ggplot2.
* Overlapping discrete axis labels raise a warning; rotating them fixes it.

Render a still of the last frame::

    manim -ql -s examples/break_followups.py BreakFollowups
"""

import numpy as np
import pandas as pd
from manim import ORIGIN, Create, Scene, VGroup, config

from ggmanim import (
    aes,
    coord_flip,
    coord_trans,
    facet_grid,
    facet_wrap,
    geom_col,
    geom_point,
    ggplot,
    labs,
    theme,
)


def make_trans_facets():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "area": rng.uniform(0, 9000, 120),
            "price": rng.uniform(1, 10, 120),
            "city": np.repeat(["A", "B", "C", "D"], 30),
        }
    )
    return (
        ggplot(df, aes("area", "price"))
        .add(geom_point(size=0.04, alpha=0.7), facet_wrap("city", ncol=4))
        .add(coord_trans(x="sqrt"), labs(title='coord_trans(x="sqrt"), thinned breaks'))
    )


def make_flip_free_x():
    df = pd.DataFrame(
        {
            "wt": [1.5, 1.8, 2.2, 2.5, 2.9, 3.1, 3.3, 3.5, 3.8, 4.1, 4.6, 5.3],
            "mpg": [33.9, 30.4, 27.3, 26.0, 21.4, 21.0, 19.7, 18.1, 15.8, 15.2, 13.3, 10.4],
            "cyl": [4, 4, 4, 4, 6, 6, 6, 6, 8, 8, 8, 8],
        }
    )
    return (
        ggplot(df, aes("wt", "mpg"))
        .add(geom_point(size=0.06), facet_grid(cols="cyl", scales="free_x"), coord_flip())
        .add(labs(title='coord_flip() + scales="free_x": mpg free per column'))
    )


def make_rotated_discrete():
    df = pd.DataFrame(
        {"model": [f"model {c}" for c in "ABCDEFGHIJ"], "sales": np.arange(10, 0, -1.0)}
    )
    return (
        ggplot(df, aes("model", "sales"))
        .add(geom_col(fill="#4C9BE8"))
        .add(theme(axis_text_x_angle=45), labs(title="10 discrete labels, rotated 45°"))
    )


def fit_frame(group: VGroup, margin: float = 0.3) -> VGroup:
    w, h = config.frame_width - 2 * margin, config.frame_height - 2 * margin
    if group.width > w or group.height > h:
        group.scale(min(w / group.width, h / group.height))
    return group.move_to(ORIGIN)


class BreakFollowups(Scene):
    def construct(self):
        small = theme(panel_width=5.0, panel_height=2.6)
        top = make_trans_facets().add(theme(panel_width=10.0, panel_height=2.4)).build()
        left = make_flip_free_x().add(small).build()
        right = make_rotated_discrete().add(small).build()
        bottom = VGroup(left, right).arrange(buff=0.6)
        grid = VGroup(top, bottom).arrange(direction=np.array([0.0, -1.0, 0.0]), buff=0.5)
        fit_frame(grid)
        self.play(*(Create(p) for p in (top, left, right)))
        self.wait()
