"""Areas, ribbons, rectangles, tiles and polygons.

Render stills of the last frame::

    manim -ql -s examples/areas_heatmap.py AreasHeatmap
    manim -ql -s examples/areas_heatmap.py Heatmap
"""

import numpy as np
import pandas as pd
from manim import ORIGIN, UP, Create, FadeIn, Scene, VGroup, config

from ggmanim import (
    aes,
    coord_flip,
    geom_area,
    geom_line,
    geom_polygon,
    geom_rect,
    geom_ribbon,
    geom_tile,
    ggplot,
    labs,
    scale_fill_viridis_c,
    theme,
)
from ggmanim.positions import POSITIONS

# geom_area stacks by default; fall back to overlapping areas while position_stack
# is not available in this checkout.
AREA_POSITION = "stack" if "stack" in POSITIONS else "identity"


def area_data() -> pd.DataFrame:
    x = np.arange(0, 13)
    frames = []
    for i, (name, base) in enumerate([("solar", 1.0), ("wind", 2.0), ("hydro", 1.5)]):
        y = base + 0.8 * np.sin(x / 2.0 + i) + 0.1 * x * (i + 1) / 3
        frames.append(pd.DataFrame({"month": x, "twh": y, "source": name}))
    return pd.concat(frames, ignore_index=True)


def make_areas():
    alpha = 1.0 if AREA_POSITION == "stack" else 0.6
    return (
        ggplot(area_data(), aes("month", "twh", fill="source"))
        .add(geom_area(position=AREA_POSITION, alpha=alpha, color="white", linewidth=1.5))
        .add(labs(x="Month", y="TWh", fill="Source", title="geom_area"))
    )


def make_ribbon():
    x = np.linspace(0, 10, 60)
    df = pd.DataFrame({"x": x, "y": np.sin(x) + x / 5})
    df["lo"], df["hi"] = df["y"] - 0.4 - x / 20, df["y"] + 0.4 + x / 20
    return (
        ggplot(df, aes("x", "y"))
        .add(
            geom_ribbon(aes(ymin="lo", ymax="hi"), fill="#4C9BE8", alpha=0.35),
            geom_ribbon(
                aes(ymin="lo", ymax="hi"),
                fill=None,
                alpha=0,
                color="#4C9BE8",
                linetype="dashed",
                linewidth=2,
            ),
            geom_line(color="#4C9BE8"),
        )
        .add(labs(title="geom_ribbon (dashed outline)"))
    )


def heatmap_data(n: int = 24) -> pd.DataFrame:
    xx, yy = np.meshgrid(np.arange(n), np.arange(n))
    z = np.sin(xx / 4.0) * np.cos(yy / 5.0) + 0.3 * np.cos((xx + yy) / 3.0)
    return pd.DataFrame({"x": xx.ravel(), "y": yy.ravel(), "z": z.ravel()})


def make_heatmap(n: int = 24):
    return (
        ggplot(heatmap_data(n), aes("x", "y", fill="z"))
        .add(geom_tile(), name="tiles")
        .add(scale_fill_viridis_c(), labs(title="geom_tile heatmap", fill="z"))
    )


def make_shapes():
    t = np.linspace(0, 2 * np.pi, 7)[:-1]
    hexagon = pd.DataFrame(
        {"x": 3 + 2 * np.cos(t), "y": 3 + 2 * np.sin(t), "id": "hex", "ring": "outer"}
    )
    hole = pd.DataFrame(
        {"x": 3 + 0.9 * np.cos(-t), "y": 3 + 0.9 * np.sin(-t), "id": "hex", "ring": "hole"}
    )
    tri = pd.DataFrame({"x": [6, 9, 7.5], "y": [1, 1, 4.5], "id": "tri", "ring": "outer"})
    polys = pd.concat([hexagon, hole, tri], ignore_index=True)
    boxes = pd.DataFrame({"x0": [0.5, 6.2], "x1": [1.5, 8.8], "y0": [5.5, 5.0], "y1": [6.5, 6.0]})
    return (
        ggplot(polys, aes("x", "y"))
        .add(
            geom_rect(
                aes(xmin="x0", xmax="x1", ymin="y0", ymax="y1"),
                data=boxes,
                inherit_aes=False,
                fill="#888888",
                color="white",
                linewidth=2,
                linetype="dashed",
            ),
            geom_polygon(aes(group="id", subgroup="ring", fill="id"), color="white", linewidth=2),
        )
        .add(labs(title="geom_polygon + geom_rect"))
    )


def make_flipped_ribbon():
    return make_ribbon().add(coord_flip(), labs(title="geom_ribbon + coord_flip"))


def fit_frame(group: VGroup, margin: float = 0.3) -> VGroup:
    w, h = config.frame_width - 2 * margin, config.frame_height - 2 * margin
    if group.width > w or group.height > h:
        group.scale(min(w / group.width, h / group.height))
    return group.move_to(ORIGIN)


class AreasHeatmap(Scene):
    def construct(self):
        small = theme(panel_width=4.2, panel_height=2.4)
        plots = [
            p.add(small).build()
            for p in (make_areas(), make_ribbon(), make_heatmap(), make_shapes())
        ]
        grid = VGroup(*plots).arrange_in_grid(rows=2, cols=2, buff=(0.6, 0.5))
        fit_frame(grid)
        self.play(*(Create(p.panel) for p in plots))
        self.play(*(FadeIn(p.layers[0], shift=UP * 0.1) for p in plots))
        self.add(grid)
        self.wait()


class Heatmap(Scene):
    def construct(self):
        plot = make_heatmap(50).build()
        self.play(Create(plot.panel))
        self.play(FadeIn(plot.layers["tiles"], lag_ratio=0.001))
        self.add(plot)
        self.wait()


class FlippedRibbon(Scene):
    def construct(self):
        plot = make_flipped_ribbon().build()
        self.add(plot)
        self.wait()
