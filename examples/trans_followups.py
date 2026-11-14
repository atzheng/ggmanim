"""coord_trans / coord_polar follow-ups (ggm-78e).

* Per-segment paths (colour varying along a line), ``geom_abline``,
  ``geom_segment`` and ``geom_curve`` are munched, so they bend with the coord.
* ``facet_grid(space="free_x")`` sizes panels by their *transformed* ranges.
* With ``clip="off"`` an area from 0 on a log axis reaches the panel edge.

Render a still of the last frame::

    manim -ql -s examples/trans_followups.py TransFollowups
"""

import numpy as np
import pandas as pd
from manim import ORIGIN, Scene, VGroup, config

from ggmanim import (
    aes,
    arrow,
    coord_polar,
    coord_trans,
    facet_grid,
    geom_abline,
    geom_area,
    geom_curve,
    geom_path,
    geom_point,
    geom_segment,
    ggplot,
    labs,
    scale_x_continuous,
    scale_y_continuous,
    theme,
)

DECADES = scale_x_continuous(breaks=[1, 10, 100, 1000])


def reference_lines(coord, title: str):
    """A colour-graded path, an abline, a segment with an arrow and a curve."""
    path = pd.DataFrame({"x": np.linspace(1, 1000, 6), "y": np.linspace(1, 1000, 6)})
    path["t"] = np.arange(len(path), dtype=float)
    arrows = pd.DataFrame({"x": [1.0], "y": [900.0], "xend": [1000.0], "yend": [100.0]})
    return (
        ggplot(path, aes("x", "y"))
        .add(geom_path(aes(colour="t"), linewidth=5))
        .add(geom_abline(slope=0.5, intercept=100, colour="#59A14F", linewidth=3))
        .add(
            geom_segment(
                aes(x="x", y="y", xend="xend", yend="yend"),
                data=arrows,
                inherit_aes=False,
                arrow=arrow(type="closed"),
                colour="#E8704C",
                linewidth=3,
            ),
            geom_curve(
                aes(x="x", y="y", xend="xend", yend="yend"),
                data=arrows,
                inherit_aes=False,
                curvature=-0.3,
                colour="#B07AA1",
                linewidth=3,
            ),
        )
        .add(coord, labs(title=title, colour="t"))
    )


def free_space():
    df = pd.DataFrame(
        {
            "x": [1.0, 10.0, 1.0, 1000.0],
            "y": [1.0, 2.0, 1.5, 2.5],
            "f": ["1 decade", "1 decade", "3 decades", "3 decades"],
        }
    )
    return (
        ggplot(df, aes("x", "y"))
        .add(geom_point(size=0.08))
        .add(
            facet_grid(cols="f", scales="free_x", space="free_x"),
            coord_trans(x="log10"),
            DECADES,
            labs(title='facet_grid(space="free_x") + coord_trans(x="log10")'),
        )
    )


def area_from_zero():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 800.0, 60.0, 300.0]})
    return (
        ggplot(df, aes("x", "y"))
        .add(geom_area(fill="#4C9BE8", alpha=0.6, colour="#4C9BE8"))
        .add(geom_segment(aes(xend="x", yend=0), colour="#E8704C", linewidth=3))
        .add(
            coord_trans(y="log10", ylim=(1, 1000), clip="off"),
            scale_y_continuous(breaks=[1, 10, 100, 1000]),
            labs(title='area from 0, coord_trans(y="log10", clip="off")'),
        )
    )


def fit_frame(group: VGroup, margin: float = 0.3) -> VGroup:
    w, h = config.frame_width - 2 * margin, config.frame_height - 2 * margin
    if group.width > w or group.height > h:
        group.scale(min(w / group.width, h / group.height))
    return group.move_to(ORIGIN)


class TransFollowups(Scene):
    def construct(self):
        small = theme(panel_width=4.2, panel_height=2.4)
        specs = [
            reference_lines(coord_trans(x="log10"), 'coord_trans(x="log10")').add(DECADES),
            reference_lines(coord_polar(), "coord_polar()"),
            free_space(),
            area_from_zero(),
        ]
        plots = [p.add(small).build() for p in specs]
        grid = VGroup(*plots).arrange_in_grid(rows=2, cols=2, buff=(0.6, 0.5))
        self.add(fit_frame(grid))
