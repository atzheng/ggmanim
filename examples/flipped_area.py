"""Horizontal stacked areas and ribbons (``orientation="y"``, ``flipped_aes``).

Render a still of the last frame::

    manim -ql -s examples/flipped_area.py FlippedArea
"""

import numpy as np
import pandas as pd
from manim import ORIGIN, Scene, VGroup, config

from ggmanim import aes, geom_area, geom_path, geom_ribbon, ggplot, labs, theme


def depth_data() -> pd.DataFrame:
    """Share of three water masses by depth; each observed at its own depths."""
    rng = np.random.default_rng(7)
    frames = []
    for i, (name, n) in enumerate([("surface", 9), ("intermediate", 7), ("deep", 11)]):
        depth = np.sort(rng.uniform(0, 1000, n))
        depth[[0, -1]] = 0, 1000
        share = 1.5 + np.sin(depth / 250 + 2 * i) + 0.3 * i
        frames.append(pd.DataFrame({"depth": depth, "share": share, "mass": name}))
    return pd.concat(frames, ignore_index=True)


def make_stacked():
    return (
        ggplot(depth_data(), aes(x="share", y="depth", fill="mass"))
        .add(geom_area(orientation="y", color="white", linewidth=1.5))
        .add(labs(x="Share", y="Depth (m)", fill="Water mass", title="geom_area(orientation='y')"))
    )


def make_ribbon():
    t = np.linspace(0, 10, 50)
    df = pd.DataFrame({"t": t, "mid": np.sin(t) + t / 4})
    df["lo"], df["hi"] = df["mid"] - 0.5, df["mid"] + 0.5 + t / 10
    return (
        ggplot(df, aes(y="t"))
        .add(
            geom_ribbon(aes(xmin="lo", xmax="hi"), fill="#4C9BE8", alpha=0.4, color="#4C9BE8"),
            geom_path(aes(x="mid"), color="#1B4F8A"),
        )
        .add(labs(x="value", y="t", title="geom_ribbon(aes(y, xmin, xmax))"))
    )


def fit_frame(group: VGroup, margin: float = 0.3) -> VGroup:
    w, h = config.frame_width - 2 * margin, config.frame_height - 2 * margin
    if group.width > w or group.height > h:
        group.scale(min(w / group.width, h / group.height))
    return group.move_to(ORIGIN)


class FlippedArea(Scene):
    def construct(self):
        small = theme(panel_width=3.6, panel_height=4.0)
        plots = [p.add(small).build() for p in (make_stacked(), make_ribbon())]
        self.add(fit_frame(VGroup(*plots).arrange(buff=0.6)))
        self.wait()
