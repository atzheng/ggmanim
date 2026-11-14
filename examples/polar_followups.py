"""coord_polar follow-ups: reference lines bend, closed paths, small panels.

* ``geom_hline``/``geom_vline``/``geom_abline`` are munched under
  ``coord_polar``: a reference line on the radius becomes a circle, one on the
  angle a ray, an abline a spiral.
* ``geom_line(closed=True)`` joins each group's last point back to its first,
  continuing round the circle: a radar chart with curved edges, no
  ``coord_radar`` needed.
* Small panels keep the theta labels clear of the radius axis and thin the
  radius breaks.

Render a still of the last frame::

    manim -ql -s examples/polar_followups.py PolarFollowups
"""

import numpy as np
import pandas as pd
from manim import ORIGIN, Create, Scene, VGroup, config

from ggmanim import (
    aes,
    coord_polar,
    geom_abline,
    geom_hline,
    geom_line,
    geom_point,
    geom_vline,
    ggplot,
    labs,
    theme,
    ylim,
)

SKILLS = ["speed", "power", "range", "defence", "stamina"]
RADAR = pd.DataFrame(
    {
        "skill": pd.Categorical(SKILLS * 2, categories=SKILLS),
        "player": ["A"] * 5 + ["B"] * 5,
        "score": [8, 5, 7, 4, 6, 4, 9, 5, 8, 7],
    }
)

WAVE = pd.DataFrame({"x": np.arange(1, 9), "y": [3, 5, 4, 7, 6, 8, 5, 4]})


def make_reference_lines():
    return (
        ggplot(WAVE, aes("x", "y"))
        .add(
            geom_line(),
            geom_hline(yintercept=5, color="#E4572E", linetype="dashed"),
            geom_vline(xintercept=[2, 6], color="#3A86C8"),
            geom_abline(slope=0.6, intercept=2, color="#3BA55C"),
        )
        .add(coord_polar(), labs(title="hline / vline / abline"))
    )


def make_closed_radar():
    return (
        ggplot(RADAR, aes("skill", "score", color="player", group="player"))
        .add(geom_line(closed=True, linewidth=3), geom_point(size=0.08))
        .add(
            coord_polar(),
            ylim(0, 10),
            labs(title="geom_line(closed=True)", x=None, y=None),
            theme(legend_position="none"),
        )
    )


def make_small():
    return (
        ggplot(WAVE, aes("x", "y"))
        .add(geom_line(closed=True), geom_hline(yintercept=6, color="#E4572E"))
        .add(coord_polar(), theme(panel_width=1.8, panel_height=1.8), labs(title="small"))
    )


def fit_frame(group: VGroup, margin: float = 0.3) -> VGroup:
    w, h = config.frame_width - 2 * margin, config.frame_height - 2 * margin
    if group.width > w or group.height > h:
        group.scale(min(w / group.width, h / group.height))
    return group.move_to(ORIGIN)


class PolarFollowups(Scene):
    def construct(self):
        size = theme(panel_width=3.6, panel_height=3.6)
        plots = [
            make_reference_lines().add(size).build(),
            make_closed_radar().add(size).build(),
            make_small().build(),
        ]
        row = VGroup(*plots).arrange(buff=0.6)
        fit_frame(row)
        self.play(*(Create(p) for p in plots))
        self.wait()
