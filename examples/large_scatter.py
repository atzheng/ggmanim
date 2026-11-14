"""Large layers: batched marks (``fast=``, ``theme.fast_threshold``).

Above ``theme.fast_threshold`` rows (2000) per panel, ``geom_point``,
``geom_tile``/``geom_rect``, ``geom_col`` and ``geom_segment`` draw one
``VMobject`` per style instead of one per row, so a 20k-point scatter builds in
well under a second. ``fast=False`` forces per-row marks, ``fast=True`` batches
small layers too.

Render stills of the last frame::

    manim -ql -s examples/large_scatter.py LargeScatter
    manim -ql -s examples/large_scatter.py FastVsSlow
"""

import time

import numpy as np
import pandas as pd
from manim import ORIGIN, Scene, VGroup, config

from ggmanim import aes, geom_point, ggplot, labs, theme


def clusters(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centres = np.array([[-2.0, -1.0], [1.5, 1.0], [0.5, -2.0]])
    k = rng.integers(0, len(centres), n)
    xy = centres[k] + rng.normal(scale=[0.9, 0.6], size=(n, 2))
    return pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "cluster": np.array(list("abc"))[k]})


def make_large(n: int = 20_000):
    return (
        ggplot(clusters(n), aes("x", "y", color="cluster"))
        .add(geom_point(size=0.025, alpha=0.35), name="pts")
        .add(labs(title=f"{n:,} points, batched (alpha stacks as per mark)"))
    )


def fit_frame(group: VGroup, margin: float = 0.3) -> VGroup:
    w, h = config.frame_width - 2 * margin, config.frame_height - 2 * margin
    if group.width > w or group.height > h:
        group.scale(min(w / group.width, h / group.height))
    return group.move_to(ORIGIN)


class LargeScatter(Scene):
    def construct(self):
        t = time.perf_counter()
        plot = make_large().build()
        print(f"built in {time.perf_counter() - t:.2f} s; batches: {len(plot.layers['pts'].marks)}")
        self.play(plot.create())
        self.wait()


class FastVsSlow(Scene):
    """The same 3000 points per row (left) and batched (right)."""

    def construct(self):
        small = theme(panel_width=4.6, panel_height=3.4, legend_position="none")
        df = clusters(3000)
        plots = []
        for fast in (False, True):
            label = "fast=True (batched)" if fast else "fast=False (one mobject per point)"
            spec = ggplot(df, aes("x", "y", color="cluster")).add(
                geom_point(size=0.04, alpha=0.5, fast=fast), labs(title=label), small
            )
            plots.append(spec.build())
        grid = VGroup(*plots).arrange(buff=0.6)
        fit_frame(grid)
        self.add(grid)
        self.wait()
