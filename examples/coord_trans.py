"""coord_trans: the coordinate system transforms the axes at drawing time.

Straight lines, polygons and rectangles in data space are *munched* (cut into
short pieces) so they bend with the transform; stats such as ``geom_smooth``
fit in data space. Render stills of the last frame::

    manim -ql -s examples/coord_trans.py CoordTrans
    manim -ql -s examples/coord_trans.py LineToCurve
"""

import numpy as np
import pandas as pd
from manim import ORIGIN, Create, Scene, Transform, VGroup, config

from ggmanim import (
    aes,
    coord_cartesian,
    coord_trans,
    geom_line,
    geom_point,
    geom_polygon,
    geom_rect,
    geom_smooth,
    ggplot,
    labs,
    scale_x_continuous,
    theme,
)

DECADES = scale_x_continuous(breaks=[1, 10, 100, 1000])


def line_rects(coord, title: str):
    """A straight data-space line over three rectangles."""
    line = pd.DataFrame({"x": [1.0, 1000.0], "y": [0.0, 1000.0]})
    rects = pd.DataFrame(
        {"x0": [1.0, 10.0, 100.0], "x1": [10.0, 100.0, 1000.0], "y1": [250.0, 500.0, 750.0]}
    )
    return (
        ggplot(line, aes("x", "y"))
        .add(
            geom_rect(
                aes(xmin="x0", xmax="x1", ymin=0, ymax="y1"),
                data=rects,
                inherit_aes=False,
                fill="#4C9BE8",
                alpha=0.5,
                color="white",
                linewidth=2,
            ),
            geom_line(color="#E8704C", linewidth=4),
        )
        .add(coord, labs(title=title))
    )


def smooth_data(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(1, 100, 60)
    return pd.DataFrame({"x": x, "y": 5 + 0.8 * x + rng.normal(0, 6, 60)})


def make_smooth():
    return (
        ggplot(smooth_data(), aes("x", "y"))
        .add(geom_point(alpha=0.7), geom_smooth(method="lm", color="#E8704C"))
        .add(
            coord_trans(x="log10"),
            scale_x_continuous(breaks=[1, 3, 10, 30, 100]),
            labs(title='lm fit, coord_trans(x="log10")'),
        )
    )


def make_triangle():
    tri = pd.DataFrame({"x": [1.0, 1000.0, 1.0], "y": [1.0, 1.0, 1000.0]})
    return (
        ggplot(tri, aes("x", "y"))
        .add(geom_polygon(fill="#59A14F", color="white", linewidth=2))
        .add(
            coord_trans(x="log10", y="sqrt"),
            DECADES,
            labs(title='polygon, coord_trans(x="log10", y="sqrt")'),
        )
    )


def fit_frame(group: VGroup, margin: float = 0.3) -> VGroup:
    w, h = config.frame_width - 2 * margin, config.frame_height - 2 * margin
    if group.width > w or group.height > h:
        group.scale(min(w / group.width, h / group.height))
    return group.move_to(ORIGIN)


class CoordTrans(Scene):
    def construct(self):
        small = theme(panel_width=4.2, panel_height=2.4)
        specs = [
            line_rects(coord_cartesian(), "coord_cartesian()"),
            line_rects(coord_trans(x="log10"), 'coord_trans(x="log10")').add(DECADES),
            make_smooth(),
            make_triangle(),
        ]
        plots = [p.add(small).build() for p in specs]
        grid = VGroup(*plots).arrange_in_grid(rows=2, cols=2, buff=(0.6, 0.5))
        fit_frame(grid)
        self.play(*(Create(p) for p in plots))
        self.wait()


class LineToCurve(Scene):
    """Morph the cartesian plot into the coord_trans one: the line bends."""

    def construct(self):
        a = line_rects(coord_cartesian(), "coord_cartesian()").build()
        b = line_rects(coord_trans(x="log10"), 'coord_trans(x="log10")').add(DECADES).build()
        self.add(a)
        self.play(Transform(a, b), run_time=2)
        self.wait()
