"""facet_grid: a rows x cols matrix of panels.

Render a still of the last frame of each scene::

    manim -ql -s examples/facet_grid.py FacetGridBasic
    manim -ql -s examples/facet_grid.py FacetGridFree
    manim -ql -s examples/facet_grid.py FacetGridSpace
    manim -ql -s examples/facet_grid.py FacetGridMargins
"""

import numpy as np
import pandas as pd
from manim import Create, FadeIn, Scene

from ggmanim import (
    aes,
    facet_grid,
    geom_col,
    geom_point,
    ggplot,
    label_both,
    labs,
    theme,
)


def cars(seed: int = 7, n: int = 180) -> pd.DataFrame:
    """A small mpg-like table: displacement vs highway mileage by drive / cylinders."""
    rng = np.random.default_rng(seed)
    drv = rng.choice(["4wd", "front", "rear"], n, p=[0.4, 0.45, 0.15])
    cyl = rng.choice([4, 6, 8], n)
    displ = np.round(1.2 + 0.55 * cyl + rng.normal(0, 0.4, n), 1)
    hwy = np.round(42 - 2.6 * displ - 3 * (drv == "4wd") + rng.normal(0, 2, n))
    return pd.DataFrame({"displ": displ, "hwy": hwy, "drv": drv, "cyl": cyl})


def make_basic():
    """2 x 3 grid: rows by drive (2 levels), columns by cylinders."""
    df = cars().query("drv != 'rear'")
    return (
        ggplot(df, aes("displ", "hwy", colour="drv"))
        .add(geom_point(size=0.05), name="points")
        .add(facet_grid(rows="drv", cols="cyl", labeller=label_both))
        .add(labs(title="facet_grid(rows='drv', cols='cyl')"))
    )


def make_free():
    """One y scale per row, row strips switched to the left."""
    return (
        ggplot(cars(), aes("displ", "hwy"))
        .add(geom_point(size=0.04), name="points")
        .add(facet_grid("drv ~ cyl", scales="free_y", switch="y"))
        .add(labs(title="scales='free_y', switch='y'"))
    )


def sales() -> pd.DataFrame:
    """Discrete x with a different number of levels per group."""
    items = {
        "fruit": ["apple", "pear", "plum", "fig", "kiwi"],
        "veg": ["leek", "kale"],
        "grain": ["oat", "rye", "rice"],
    }
    rows = []
    for k, (kind, names) in enumerate(items.items()):
        for j, name in enumerate(names):
            for year in (2025, 2026):
                rows.append((kind, name, year, 3 + (k * 5 + j * 3 + year) % 7))
    return pd.DataFrame(rows, columns=["kind", "item", "year", "sold"])


def make_space():
    """space='free_x': every bar has the same width across the columns."""
    return (
        ggplot(sales(), aes("item", "sold"))
        .add(geom_col(fill="#56B4E9"), name="bars")
        .add(facet_grid("year ~ kind", scales="free_x", space="free_x"))
        .add(labs(title="scales='free_x', space='free_x'"))
    )


def make_margins():
    df = cars().query("drv != 'rear'")
    return (
        ggplot(df, aes("displ", "hwy"))
        .add(geom_point(size=0.04, alpha=0.7), name="points")
        .add(facet_grid("drv ~ cyl", margins=True))
        .add(labs(title="margins=True"))
    )


class _GridScene(Scene):
    plot = staticmethod(make_basic)
    layer = "points"

    def construct(self):
        built = self.plot().add(theme(panel_width=10.5, panel_height=5.6)).build()
        built.move_to([0, 0, 0])
        self.play(FadeIn(built.panel_grid))
        self.play(Create(built.layers[self.layer]))
        self.add(built)
        self.wait()


class FacetGridBasic(_GridScene):
    plot = staticmethod(make_basic)


class FacetGridFree(_GridScene):
    plot = staticmethod(make_free)


class FacetGridSpace(_GridScene):
    plot = staticmethod(make_space)
    layer = "bars"


class FacetGridMargins(_GridScene):
    plot = staticmethod(make_margins)
