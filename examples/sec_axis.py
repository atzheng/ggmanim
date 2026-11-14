"""Secondary axes: ``sec_axis`` / ``dup_axis`` on the opposite side of the panel.

Render stills of the last frame::

    manim -ql -s examples/sec_axis.py SecAxis
    manim -ql -s examples/sec_axis.py SecAxisFacets
"""

import numpy as np
import pandas as pd
from manim import ORIGIN, Create, Scene, VGroup, config

from ggmanim import (
    aes,
    coord_flip,
    dup_axis,
    facet_wrap,
    geom_col,
    geom_line,
    geom_point,
    ggplot,
    labs,
    scale_x_continuous,
    scale_x_log10,
    scale_y_continuous,
    sec_axis,
    theme,
)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def temps() -> pd.DataFrame:
    m = np.arange(1, 13)
    return pd.DataFrame({"month": m, "celsius": 9 + 11 * np.sin((m - 4) / 12 * 2 * np.pi)})


def fahrenheit():
    return scale_y_continuous(
        name="°C", sec_axis=sec_axis(lambda c: c * 1.8 + 32, name="°F", breaks=[32, 50, 68])
    )


def make_celsius():
    return (
        ggplot(temps(), aes("month", "celsius"))
        .add(geom_line(color="#E8704C", linewidth=4), geom_point(color="#E8704C"))
        .add(fahrenheit(), labs(x="month", title="sec_axis: °C left, °F right"))
    )


def make_log():
    rng = np.random.default_rng(1)
    mass = 10 ** rng.uniform(-1, 3.5, 40)
    df = pd.DataFrame({"mass": mass, "rate": 3.4 * mass**0.75 * rng.lognormal(0, 0.2, 40)})
    return (
        ggplot(df, aes("mass", "rate"))
        .add(geom_point(color="#4C9BE8", alpha=0.8))
        .add(scale_x_log10(name="mass (kg)", sec_axis=dup_axis()))
        .add(labs(title="scale_x_log10 + dup_axis()"))
    )


def make_flip():
    df = pd.DataFrame({"city": ["Oslo", "Lima", "Pune", "Kyiv"], "celsius": [6.0, 19.4, 25.2, 8.4]})
    return (
        ggplot(df, aes("city", "celsius"))
        .add(geom_col(fill="#59A14F", width=0.6))
        .add(
            scale_y_continuous(
                name="mean °C", sec_axis=sec_axis(lambda c: c * 1.8 + 32, name="mean °F")
            ),
            coord_flip(),
            labs(title="coord_flip: °F on top"),
        )
    )


def make_ratio():
    x = np.linspace(0, 10, 40)
    df = pd.DataFrame({"x": x, "y": 100 * (1 - np.exp(-x / 3))})
    return (
        ggplot(df, aes("x", "y"))
        .add(geom_line(color="#B07AA1", linewidth=4))
        .add(
            scale_y_continuous(
                name="percent",
                sec_axis=sec_axis(lambda p: p / 100, name="fraction"),
            ),
            labs(title="secondary labels: primary formatter"),
        )
    )


def months_axis():
    """Quarterly month ticks, mirrored on top of the facet grid with ``dup_axis``."""
    return scale_x_continuous(
        breaks=[1, 4, 7, 10], labels=[MONTHS[i - 1] for i in (1, 4, 7, 10)], sec_axis=dup_axis()
    )


def fit_frame(group: VGroup, margin: float = 0.3) -> VGroup:
    w, h = config.frame_width - 2 * margin, config.frame_height - 2 * margin
    if group.width > w or group.height > h:
        group.scale(min(w / group.width, h / group.height))
    return group.move_to(ORIGIN)


class SecAxis(Scene):
    def construct(self):
        small = theme(panel_width=4.0, panel_height=2.2)
        plots = [
            p.add(small).build() for p in (make_celsius(), make_log(), make_flip(), make_ratio())
        ]
        grid = VGroup(*plots).arrange_in_grid(rows=2, cols=2, buff=(0.6, 0.5))
        fit_frame(grid)
        self.play(*(Create(p) for p in plots))
        self.wait()


class SecAxisFacets(Scene):
    """Faceted: the secondary axis is drawn on the outer (right / top) panels only."""

    def construct(self):
        rng = np.random.default_rng(4)
        rows = []
        for city, base in (("Oslo", 6), ("Lima", 19), ("Pune", 25), ("Kyiv", 8), ("Rome", 15)):
            m = np.arange(1, 13)
            c = base + 9 * np.sin((m - 4) / 12 * 2 * np.pi) + rng.normal(0, 1, 12)
            rows.append(pd.DataFrame({"city": city, "month": m, "celsius": c}))
        df = pd.concat(rows, ignore_index=True)
        plot = (
            ggplot(df, aes("month", "celsius"))
            .add(geom_line(color="#E8704C", linewidth=3))
            .add(
                facet_wrap("city", ncol=3),
                fahrenheit(),
                months_axis(),
                labs(title="facet_wrap + sec_axis"),
            )
            .build()
        )
        fit_frame(plot)
        self.play(Create(plot))
        self.wait()
