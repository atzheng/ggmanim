"""Secondary axes on date / date-time scales, and the top/right theme switches.

Render a still of the last frame::

    manim -ql -s examples/sec_axis_date.py SecAxisDate
"""

import numpy as np
import pandas as pd
from manim import ORIGIN, Create, Scene, VGroup, config

from ggmanim import (
    aes,
    dup_axis,
    geom_line,
    geom_point,
    ggplot,
    label_date,
    labs,
    scale_x_date,
    scale_x_datetime,
    scale_y_continuous,
    sec_axis,
    theme,
)


def make_timezones():
    """One day of readings in UTC, with Tokyo time on top (``dup_axis`` + ``label_date``)."""
    t = pd.date_range("2024-03-01", periods=25, freq="h")
    load = 50 + 30 * np.sin((np.arange(25) - 9) / 24 * 2 * np.pi)
    return (
        ggplot(pd.DataFrame({"t": t, "load": load}), aes("t", "load"))
        .add(geom_line(color="#4C9BE8", linewidth=4))
        .add(
            scale_x_datetime(
                name="UTC",
                date_breaks="6 hours",
                date_labels="%H:%M",
                sec_axis=dup_axis(name="Tokyo", labels=label_date("%H:%M", tz="Asia/Tokyo")),
            ),
            labs(y="load (%)", title="scale_x_datetime + dup_axis in another zone"),
        )
    )


def make_day_of_year():
    """Monthly values: month names below, day of the year above."""
    d = pd.date_range("2024-01-01", periods=12, freq="MS")
    hours = 12 + 6 * np.sin((np.arange(12) - 2.7) / 12 * 2 * np.pi)
    return (
        ggplot(pd.DataFrame({"d": d, "hours": hours}), aes("d", "hours"))
        .add(geom_line(color="#E8704C", linewidth=4), geom_point(color="#E8704C"))
        .add(
            scale_x_date(
                name="month",
                date_breaks="3 months",
                date_labels="%b",
                sec_axis=dup_axis(name="day of year", labels=label_date("%j")),
            ),
            labs(y="daylight (h)", title="scale_x_date + dup_axis(labels=%j)"),
        )
    )


def make_switches():
    """``axis_text_y_right=False`` blanks the right labels only; the title stays."""
    t = pd.date_range("2024-03-01", periods=10, freq="D")
    c = np.array([4.0, 6, 9, 7, 11, 13, 12, 15, 14, 17])
    return (
        ggplot(pd.DataFrame({"t": t, "c": c}), aes("t", "c"))
        .add(geom_line(color="#59A14F", linewidth=4))
        .add(
            scale_x_date(
                date_labels="%d",
                name="March",
                sec_axis=sec_axis(
                    lambda d: d + 7,  # days since the epoch: one week later
                    name="a week later",
                    breaks=["2024-03-09", "2024-03-12", "2024-03-15"],
                    labels=label_date("%b %d"),
                ),
            ),
            scale_y_continuous(name="°C", sec_axis=sec_axis(lambda v: v * 1.8 + 32, name="°F")),
            theme(axis_text_y_right=False, axis_title_x_top=True),
            labs(title="axis_text_y_right=False"),
        )
    )


def fit_frame(group: VGroup, margin: float = 0.3) -> VGroup:
    w, h = config.frame_width - 2 * margin, config.frame_height - 2 * margin
    if group.width > w or group.height > h:
        group.scale(min(w / group.width, h / group.height))
    return group.move_to(ORIGIN)


class SecAxisDate(Scene):
    def construct(self):
        small = theme(panel_width=4.0, panel_height=2.2)
        plots = [p.add(small).build() for p in (make_timezones(), make_day_of_year())]
        top = VGroup(*plots).arrange(buff=0.8)
        bottom = make_switches().add(small).build()
        page = VGroup(top, bottom).arrange(direction=np.array([0, -1, 0]), buff=0.6)
        fit_frame(page)
        self.play(*(Create(p) for p in [*plots, bottom]))
        self.wait()
