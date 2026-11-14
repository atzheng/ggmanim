"""coord_polar: pie, donut, coxcomb (rose) and radar charts.

Bars are munched into wedges; ``theta="y"`` maps the stacked counts to the
angle. Render stills of the last frame::

    manim -ql -s examples/polar.py Polar
    manim -ql -s examples/polar.py BarsToPie
"""

import numpy as np
import pandas as pd
from manim import ORIGIN, Create, Scene, Transform, VGroup, config

from ggmanim import (
    aes,
    coord_cartesian,
    coord_polar,
    coord_radar,
    geom_bar,
    geom_col,
    geom_point,
    geom_polygon,
    ggplot,
    labs,
    theme,
    theme_void,
    xlim,
    ylim,
)

FRUIT = pd.DataFrame({"fruit": ["apple"] * 6 + ["banana"] * 3 + ["cherry"] * 2 + ["kiwi"] * 1})

SHARES = pd.DataFrame(
    {"x": 2, "browser": ["Chrome", "Safari", "Edge", "Firefox"], "share": [65, 19, 9, 7]}
)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def rose_data(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 20 + 15 * np.sin(np.linspace(0, 2 * np.pi, 12, endpoint=False))
    return pd.DataFrame(
        {
            "month": pd.Categorical(MONTHS * 2, categories=MONTHS),
            "cause": ["disease"] * 12 + ["wounds"] * 12,
            "deaths": np.concatenate([base + rng.uniform(0, 8, 12), 0.4 * base]),
        }
    )


RADAR = pd.DataFrame(
    {
        "skill": pd.Categorical(
            ["speed", "power", "range", "defence", "stamina"] * 2,
            categories=["speed", "power", "range", "defence", "stamina"],
        ),
        "player": ["A"] * 5 + ["B"] * 5,
        "score": [8, 5, 7, 4, 6, 4, 9, 5, 8, 7],
    }
)


def make_pie():
    return (
        ggplot(FRUIT, aes(x=1, fill="fruit"))
        .add(geom_bar(width=1, color="white", linewidth=2))
        .add(coord_polar(theta="y"), theme_void(), labs(title="pie: geom_bar + theta='y'"))
    )


def make_donut():
    return (
        ggplot(SHARES, aes("x", "share", fill="browser"))
        .add(geom_col(width=1, color="white", linewidth=2))
        .add(
            coord_polar(theta="y"),
            xlim(0.5, 2.5),
            theme_void(),
            labs(title="donut: xlim(0.5, 2.5)"),
        )
    )


def make_rose():
    return (
        ggplot(rose_data(), aes("month", "deaths", fill="cause"))
        .add(geom_col(width=1, color="white", linewidth=1))
        .add(coord_polar(), labs(title="coxcomb / rose", x=None, y=None))
    )


def make_radar():
    return (
        ggplot(RADAR, aes("skill", "score", color="player", group="player"))
        .add(geom_polygon(aes(fill="player"), alpha=0.2, linewidth=3), geom_point(size=0.08))
        .add(coord_radar(), ylim(0, 10), labs(title="radar: coord_radar()", x=None, y=None))
    )


def fit_frame(group: VGroup, margin: float = 0.3) -> VGroup:
    w, h = config.frame_width - 2 * margin, config.frame_height - 2 * margin
    if group.width > w or group.height > h:
        group.scale(min(w / group.width, h / group.height))
    return group.move_to(ORIGIN)


class Polar(Scene):
    def construct(self):
        small = theme(panel_width=3.2, panel_height=3.2)
        specs = [make_pie(), make_donut(), make_rose(), make_radar()]
        plots = [p.add(small).build() for p in specs]
        grid = VGroup(*plots).arrange_in_grid(rows=2, cols=2, buff=(0.6, 0.4))
        fit_frame(grid)
        self.play(*(Create(p) for p in plots))
        self.wait()


class BarsToPie(Scene):
    """Morph a stacked bar into a pie: the same wedges, unrolled."""

    def construct(self):
        bar = ggplot(FRUIT, aes(x=1, fill="fruit")).add(geom_bar(width=1), coord_cartesian())
        a = bar.add(labs(title="stacked bar")).build()
        b = bar.add(coord_polar(theta="y"), labs(title="coord_polar(theta='y')")).build()
        self.add(a)
        self.play(Transform(a, b), run_time=2)
        self.wait()
