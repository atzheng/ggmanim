"""Phase 4 gallery: facets.

Render a still of every scene (the last frame) into ``media/images/gallery_facets/``::

    manim -ql -s -a examples/gallery_facets.py

or one scene::

    manim -ql -s examples/gallery_facets.py PanelReveal

The data sets come from the per-feature examples (``examples/facet_wrap.py``,
``examples/facet_grid.py``). Each scene builds its plot with a ``make_*``
function, so the plots can also be built (without rendering) from tests or a
notebook.
"""

import sys
from pathlib import Path

from manim import Create, FadeIn, Scene, Write

# The per-feature examples live next to this file; manim puts this directory on
# sys.path when it renders, importing from a test or a notebook may not.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from facet_grid import cars  # noqa: E402
from facet_grid import sales as shop_sales  # noqa: E402
from facet_wrap import make_plot as make_wrap_plot  # noqa: E402
from facet_wrap import sales  # noqa: E402

from ggmanim import (  # noqa: E402
    aes,
    breaks_width,
    facet_grid,
    facet_wrap,
    geom_hline,
    geom_point,
    geom_smooth,
    ggplot,
    label_both,
    labs,
    scale_x_continuous,
    theme,
)

#: Panel-grid size used by every scene: nearly the whole 14 x 8 frame.
WIDE = theme(panel_width=10.5, panel_height=5.2)


# -- G1: facet_wrap, fixed scales ------------------------------------------------------


def make_wrap_fixed():
    """Seven weekdays wrapped into 2 rows of 4; one shared x and y scale.

    Four narrow columns: x breaks every 4 hours keep the tick labels apart."""
    return make_wrap_plot(ncol=4).add(
        scale_x_continuous(breaks=breaks_width(4)),
        WIDE,
        labs(title="facet_wrap('day', ncol=4)"),
    )


class WrapFixed(Scene):
    def construct(self):
        plot = make_wrap_fixed().build()
        self.play(FadeIn(plot.panel_grid))
        self.play(FadeIn(plot.layers["points"], lag_ratio=0.01))
        self.play(Create(plot.layers["fit"]))
        self.add(plot)
        self.wait()


# -- G2: facet_wrap, free y scales -----------------------------------------------------


def make_wrap_free_y():
    """Four days in a 2 x 2 grid, each with its own y axis (weekends sell ~3x)."""
    df = sales()
    df = df[df["day"].isin(["Mon", "Thu", "Sat", "Sun"])]
    return (
        ggplot(df, aes("hour", "units"))
        .add(geom_point(alpha=0.8, size=0.05), name="points")
        .add(geom_smooth(method="lm", se=False), name="fit")
        .add(
            facet_wrap("day", ncol=2, scales="free_y", labeller=label_both),
            labs(x="Hour", y="Units sold", title="scales='free_y', label_both"),
            WIDE,
        )
    )


class WrapFreeY(Scene):
    def construct(self):
        plot = make_wrap_free_y().build()
        self.play(FadeIn(plot.panel_grid))
        self.play(FadeIn(plot.layers["points"], lag_ratio=0.01))
        self.play(Create(plot.layers["fit"]))
        self.add(plot)
        self.wait()


# -- G3: facet_grid with switched strips -----------------------------------------------


def make_grid_switch():
    """drive x cylinders; strips switched to the bottom / left, outside the axes."""
    df = cars().query("drv != 'rear'")
    return (
        ggplot(df, aes("displ", "hwy", colour="drv"))
        .add(geom_point(size=0.05), name="points")
        .add(
            facet_grid("drv ~ cyl", switch="both", labeller=label_both),
            theme(strip_placement="outside", legend_position="none"),
            labs(x="Displacement (l)", y="Highway mpg", title="switch='both'"),
            WIDE,
        )
    )


class GridSwitch(Scene):
    def construct(self):
        plot = make_grid_switch().build()
        self.play(FadeIn(plot.panel_grid))
        self.play(FadeIn(plot.layers["points"], lag_ratio=0.01))
        self.add(plot)
        self.wait()


# -- G4: facet_grid with free space ----------------------------------------------------


def make_grid_space():
    """A dot plot by product kind: each row is as tall as its number of items."""
    df = shop_sales().assign(year=lambda d: d["year"].astype(str))
    return (
        ggplot(df, aes("sold", "item", colour="year"))
        .add(geom_point(size=0.08), name="points")
        .add(
            facet_grid(rows="kind", scales="free_y", space="free_y"),
            labs(x="Units sold", y=None, colour="Year",
                 title="scales='free_y', space='free_y'"),
            WIDE,
        )
    )  # fmt: skip


class GridSpace(Scene):
    def construct(self):
        plot = make_grid_space().build()
        self.play(FadeIn(plot.panel_grid))
        self.play(FadeIn(plot.layers["points"], lag_ratio=0.05))
        self.add(plot)
        self.wait()


# -- G5: a non-faceted reference layer, repeated in every panel ------------------------


def make_reference_layer():
    """Every car in grey behind each drive's own cars; the overall mean as a line.

    The grey layer's data has no ``drv`` column, so (as in ggplot2) it is repeated
    in every panel; so is ``geom_hline``, which has no data at all.
    """
    df = cars()
    everyone = df.drop(columns="drv")
    return (
        ggplot(df, aes("displ", "hwy"))
        .add(geom_point(data=everyone, colour="#777777", alpha=0.35, size=0.04),
             name="all")
        .add(geom_hline(yintercept=float(df["hwy"].mean()), linetype="dashed"),
             name="mean")
        .add(geom_point(aes(colour="drv"), size=0.06), name="points")
        .add(
            facet_wrap("drv"),
            labs(x="Displacement (l)", y="Highway mpg",
                 title="Grey: every car (no drv column, so in every panel)"),
            theme(legend_position="none"),
            WIDE,
        )
    )  # fmt: skip


class ReferenceLayer(Scene):
    def construct(self):
        plot = make_reference_layer().build()
        self.play(FadeIn(plot.panel_grid))
        self.play(FadeIn(plot.layers["all"]), Create(plot.layers["mean"]))
        self.play(FadeIn(plot.layers["points"], lag_ratio=0.01))
        self.add(plot)
        self.wait()


# -- G6: animation, one panel at a time ------------------------------------------------


class PanelReveal(Scene):
    """Reveal the wrapped panels one by one via ``GGPlot.panels[(row, col)]``,
    each with its strip and its own slice of every layer."""

    def construct(self):
        plot = make_wrap_fixed().build()
        grid = plot.panel_grid
        self.play(Write(plot.title), FadeIn(grid.x_title), FadeIn(grid.y_title))
        for (row, col), panel in plot.panels.items():
            self.play(
                FadeIn(panel),
                FadeIn(grid.panel_strips[(row, col)]),
                run_time=0.5,
            )
            self.play(
                FadeIn(plot.layers["points"].panel(row, col), lag_ratio=0.05),
                Create(plot.layers["fit"].panel(row, col)),
                run_time=0.6,
            )
        self.add(plot)
        self.wait()
