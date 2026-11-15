"""Transitions on date and discrete axes, stable groups, rescaling axes.

Render (``-s`` gives only the last frame)::

    manim -ql examples/transition_followups.py SeriesStates    # date axis, lines keep their ids
    manim -ql examples/transition_followups.py SharedLevels    # discrete axis, union of levels
    manim -ql examples/transition_followups.py AxisRescale     # ticks fade at the panel edge
"""

import numpy as np
import pandas as pd
from manim import Scene

from ggmanim import animate_states, geom_col, geom_line, geom_point, ggplot, labs, transition
from ggmanim.aes import aes


def series() -> pd.DataFrame:
    """Monthly values of three series over three years; "north" only starts in
    the second year and "south" stops after it."""
    rng = np.random.default_rng(3)
    rows = []
    for year in (2021, 2022, 2023):
        months = pd.date_range(f"{year}-01-01", periods=12, freq="MS")
        for name, level in (("north", 40.0), ("east", 25.0), ("south", 10.0)):
            if (name == "north" and year == 2021) or (name == "south" and year == 2023):
                continue
            walk = level + (year - 2021) * 8 + np.cumsum(rng.normal(0, 2.0, 12))
            rows.append(
                pd.DataFrame({"month": months, "value": walk, "region": name, "year": year})
            )
    return pd.concat(rows, ignore_index=True)


class SeriesStates(Scene):
    gallery_gif = True  # render as a GIF in the docs gallery

    """One state per year on a date axis: the shared x range is the union of all
    years and each region's line morphs into itself (stable group ids)."""

    def construct(self):
        spec = (
            ggplot(series(), aes("month", "value", colour="region"))
            .add(geom_line(linewidth=3), name="lines")
            .add(labs(x=None, y="Value"))
        )
        anim = animate_states(
            spec, "year", title="Year {state}", transition_length=1.2, state_length=0.5
        )
        self.play(anim)


def fruit(names, values):
    return (
        ggplot(pd.DataFrame({"fruit": names, "n": values}), aes("fruit", "n"))
        .add(geom_col(fill="#4C9BE8"), name="bars")
        .add(labs(x=None, y="Count"))
    )


class SharedLevels(Scene):
    gallery_gif = True  # render as a GIF in the docs gallery

    """A discrete x axis shared by both plots (``limits="union"``): bars that
    stay keep their place, new ones grow in their own slot."""

    def construct(self):
        a = fruit(["apple", "banana", "cherry"], [3, 5, 2])
        b = fruit(["banana", "cherry", "date", "fig"], [6, 1, 4, 3])
        t = transition(a, b, key="fruit", limits="union", run_time=1.5)
        self.add(t.plot_a)
        self.wait(0.3)
        self.play(t)
        # GGPlot.transition_to, keeping this plot's levels
        self.play(t.plot.transition_to(fruit(["apple", "fig"], [4, 2]), key="fruit", limits="a"))
        self.wait(0.3)


class AxisRescale(Scene):
    gallery_gif = True  # render as a GIF in the docs gallery

    """The y axis grows twenty-fold: incoming labels appear at the top edge one
    at a time instead of piling up there."""

    def construct(self):
        rng = np.random.default_rng(5)
        x = rng.uniform(0, 10, 30)
        small = pd.DataFrame({"x": x, "y": rng.uniform(0, 5, 30), "id": range(30)})
        big = small.assign(y=small.y * rng.uniform(1, 20, 30))
        spec = lambda d: ggplot(d, aes("x", "y")).add(geom_point(size=0.08), name="pts")  # noqa: E731
        t = transition(spec(small), spec(big), key="id", run_time=2)
        self.add(t.plot_a)
        self.wait(0.3)
        self.play(t)
        self.play(transition(t.plot, spec(small), key="id", run_time=2))
        self.wait(0.3)
