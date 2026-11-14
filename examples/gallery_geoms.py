"""Phase 3 gallery: geoms, stats and positions, one scene per idea.

The plots come from the per-feature examples (``histograms.py``,
``distributions.py``, ``areas_heatmap.py``, ``annotations.py``, ``smoothers.py``,
``two_d.py``); this file imports their ``make_*`` builders and data helpers and only
adds the few plots and layers that combine them (stacked count labels, histogram +
density, step + error bars, the annotate() rect and arrow). ``GeomsOverview`` puts
nine of them on one frame.

Render stills of every scene::

    manim -ql -s -a examples/gallery_geoms.py

or one scene::

    manim -ql -s examples/gallery_geoms.py HistogramDensity
"""

import numpy as np
import pandas as pd

# annotations, areas_heatmap, distributions, histograms, smoothers and two_d are the
# sibling example files (manim puts this directory on sys.path when it loads the
# file). Only builders are imported, never Scene classes, so ``-a`` renders just the
# scenes defined here.
from annotations import make_labelled_scatter
from areas_heatmap import make_areas, make_heatmap
from distributions import make_gallery as make_violin_box_jitter
from histograms import diamonds_like, heights
from histograms import make_plots as make_histogram_plots
from manim import ORIGIN, UP, Create, FadeIn, Scene, VGroup, config
from smoothers import make_smooths
from two_d import make_density

from ggmanim import (
    aes,
    after_stat,
    annotate,
    arrow,
    geom_bar,
    geom_density,
    geom_histogram,
    geom_text,
    ggplot,
    labs,
    position_stack,
    stat_summary,
    theme,
)


def fit_frame(group: VGroup, margin: float = 0.3) -> VGroup:
    """Shrink ``group`` (never enlarge it) to fit the frame, and centre it."""
    w, h = config.frame_width - 2 * margin, config.frame_height - 2 * margin
    if group.width > w or group.height > h:
        group.scale(min(w / group.width, h / group.height))
    return group.move_to(ORIGIN)


# -- 1. stacked and dodged bars, counts labelled --------------------------------------


def make_stacked_counts():
    """geom_bar() stacks counts; the labels stack with position_stack(vjust=0.5)."""
    return (
        ggplot(diamonds_like(), aes("cut", fill="clarity"))
        .add(geom_bar(), name="bars")
        .add(
            geom_text(
                aes(label=after_stat("count")),
                stat="count",
                position=position_stack(vjust=0.5),
                size=2.5,
                color="white",
                show_legend=False,
            ),
            name="counts",
        )
        .add(labs(title="geom_bar() stacked + counts", x=None, fill="Clarity"))
    )


def make_dodged_counts():
    """The dodged twin from ``histograms.py`` (counts above each bar)."""
    return make_histogram_plots()[3]


# -- 2. histogram with a density overlay ----------------------------------------------


def make_histogram_density():
    return (
        ggplot(heights(), aes("height"))
        .add(
            geom_histogram(
                aes(y=after_stat("density"), fill="sex"),
                binwidth=2.5,
                alpha=0.5,
                position="identity",
            ),
            name="hist",
        )
        # fill is mapped on the histogram only: a density inheriting it would be filled
        .add(geom_density(aes(color="sex"), linewidth=3), name="density")
        .add(labs(title="histogram + geom_density", x="Height (cm)", fill="Sex", color="Sex"))
    )


# -- 3. step chart with error bars ----------------------------------------------------


def weekly(seed: int = 4) -> pd.DataFrame:
    """Eight weeks, ten noisy measurements a week, a staircase trend."""
    rng = np.random.default_rng(seed)
    week = np.repeat(np.arange(1, 9), 10)
    level = np.array([2.0, 2.0, 3.5, 3.5, 3.0, 5.0, 5.5, 6.5])[week - 1]
    return pd.DataFrame({"week": week, "value": level + rng.normal(0, 0.8, len(week))})


def make_step_errorbars():
    return (
        ggplot(weekly(), aes("week", "value"))
        .add(stat_summary(fun="mean", geom="step", direction="mid"), name="step")
        .add(
            stat_summary(fun_data="mean_se", geom="errorbar", width=0.3, color="#E69F00"),
            name="errorbars",
        )
        .add(stat_summary(fun="mean", geom="point", size=0.06), name="means")
        .add(labs(title='geom_step("mid") + mean ± SE', x="Week", y="Value"))
    )


# -- 4. annotated scatter -------------------------------------------------------------


def make_annotated_scatter():
    """``annotations.py``'s labelled scatter plus an annotate() rect and arrow."""
    return (
        make_labelled_scatter()
        .add(
            annotate(
                "rect",
                xmin=1.4,
                xmax=2.0,
                ymin=28.5,
                ymax=35.5,
                alpha=0,  # alpha is the fill's: an outline-only highlight box
                color="#0072B2",
                linetype="dashed",
            ),
            name="light",
        )
        .add(
            annotate(
                "segment",
                x=2.7,
                y=29.5,
                xend=2.05,
                yend=31.5,
                arrow=arrow(),
                color="#0072B2",
            ),
            name="arrow",
        )
        .add(
            annotate("text", x=2.75, y=29.5, label="light & frugal", hjust=0, color="#0072B2"),
            name="arrow_note",
        )
    )


# -- the scenes -----------------------------------------------------------------------

# (title, builder) for the overview's panels, in gallery order.
PANELS = [
    ("stacked bars + counts", make_stacked_counts),
    ("histogram + density", make_histogram_density),
    ("violin + boxplot + jitter", make_violin_box_jitter),
    ("step + error bars", make_step_errorbars),
    ("stacked area", make_areas),
    ("heatmap", make_heatmap),
    ("annotated scatter", make_annotated_scatter),
    ("loess vs lm", make_smooths),
    ("2-D density contours", make_density),
]


class _OnePlot:
    """Scene mixin: build ``make()``, draw the panel, then fade in each layer in order.

    Not a ``Scene`` itself, so ``manim -a`` does not try to render it."""

    @staticmethod
    def make():
        raise NotImplementedError

    def construct(self):
        plot = self.make().build()
        self.play(Create(plot.panel))
        for layer in plot.layers.values():
            self.play(FadeIn(layer, shift=UP * 0.1, lag_ratio=0.02))
        self.add(plot)
        self.wait()


class BarsWithCounts(Scene):
    def construct(self):
        small = theme(panel_width=4.0, panel_height=2.6, legend_position="bottom")
        plots = [m().add(small).build() for m in (make_stacked_counts, make_dodged_counts)]
        row = fit_frame(VGroup(*plots).arrange(buff=0.5))
        self.play(*(Create(p.panel) for p in plots))
        self.play(*(FadeIn(p.layers[0], shift=UP * 0.1, lag_ratio=0.05) for p in plots))
        self.add(row)
        self.wait()


class HistogramDensity(_OnePlot, Scene):
    make = staticmethod(make_histogram_density)


class ViolinBoxJitter(_OnePlot, Scene):
    make = staticmethod(make_violin_box_jitter)


class StepErrorbars(_OnePlot, Scene):
    make = staticmethod(make_step_errorbars)


class StackedArea(_OnePlot, Scene):
    make = staticmethod(make_areas)


class TileHeatmap(_OnePlot, Scene):
    make = staticmethod(make_heatmap)


class AnnotatedScatter(_OnePlot, Scene):
    make = staticmethod(make_annotated_scatter)


class LoessVsLm(_OnePlot, Scene):
    make = staticmethod(make_smooths)


class Density2D(_OnePlot, Scene):
    make = staticmethod(make_density)


class GeomsOverview(Scene):
    """Every panel, small, on one frame."""

    def construct(self):
        small = theme(panel_width=4.2, panel_height=2.0, legend_position="none")
        plots = [make().add(small).build() for _, make in PANELS]
        grid = fit_frame(VGroup(*plots).arrange_in_grid(rows=3, cols=3, buff=(0.4, 0.3)), 0.15)
        self.play(FadeIn(grid, lag_ratio=0.05))
        self.wait()


SCENES = [
    BarsWithCounts,
    HistogramDensity,
    ViolinBoxJitter,
    StepErrorbars,
    StackedArea,
    TileHeatmap,
    AnnotatedScatter,
    LoessVsLm,
    Density2D,
    GeomsOverview,
]
