"""Text and annotations: counts on bars, labelled points, annotate(), direct labels.

Render stills of every scene::

    manim -ql -s -a examples/annotations.py
"""

import numpy as np
import pandas as pd
from manim import DOWN, UP, Create, FadeIn, Scene, Write

from ggmanim import (
    aes,
    annotate,
    geom_col,
    geom_label,
    geom_line,
    geom_point,
    geom_text,
    geom_vline,
    ggplot,
    labs,
    theme,
)


def make_counts_on_bars():
    """Bar heights written above each bar (``vjust=0`` + ``nudge_y``)."""
    levels = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    counts = pd.DataFrame({"cut": pd.Categorical(levels, categories=levels)})
    counts["n"] = [16, 49, 121, 138, 215]
    return (
        ggplot(counts, aes("cut", "n"))
        .add(geom_col(fill="#4C72B0"), name="bars")
        .add(geom_text(aes(label="n"), vjust=0, nudge_y=4), name="counts")
        .add(labs(title="Counts on bars", x="Cut", y="Diamonds"))
    )


def make_labelled_scatter():
    """geom_label on a few points, annotate() for text, a highlight and a vline."""
    cars = pd.DataFrame(
        {
            "model": ["Mazda", "Fiat", "Honda", "Toyota", "Camaro", "Lotus", "Volvo", "Merc"],
            "wt": [2.62, 2.20, 1.62, 1.84, 3.84, 1.51, 2.78, 3.44],
            "mpg": [21.0, 32.4, 30.4, 33.9, 13.3, 30.4, 21.4, 19.2],
        }
    )
    best = cars.nlargest(3, "mpg")
    return (
        ggplot(cars, aes("wt", "mpg"))
        .add(geom_point(size=0.07), name="points")
        .add(
            geom_label(aes(label="model"), data=best, nudge_x=0.06, hjust=0, fill="#FDF6E3"),
            name="best",
        )
        .add(geom_vline(xintercept=3.0, linetype="dashed", label="3 tons"), name="cutoff")
        .add(
            annotate("point", x=3.84, y=13.3, size=0.14, color="#E24A33", alpha=0.6),
            name="highlight",
        )
        .add(
            annotate(
                "text",
                x=3.84,
                y=15.0,
                label="heaviest",
                color="#E24A33",
                fontface="italic",
                vjust=0,
            ),
            name="note",
        )
        .add(labs(title="Labels and annotations", x="Weight (1000 lbs)", y="MPG"))
    )


def make_direct_labels():
    """Direct labels at line ends instead of a legend, plus angled text."""
    x = np.linspace(0, 10, 41)
    df = pd.concat(
        [pd.DataFrame({"x": x, "y": f(x), "series": name}) for name, f in SERIES.items()],
        ignore_index=True,
    )
    return (
        ggplot(df, aes("x", "y", color="series"))
        .add(geom_line(label=sorted(SERIES)), name="lines")
        .add(
            annotate("text", x=7, y=1.6, label="steady growth", angle=14, size=5, alpha=0.6),
            name="note",
        )
        .add(labs(title="Direct labels", x="t", y="value"))
        .add(theme(legend_position="none"))
    )


SERIES = {
    "linear": lambda x: 0.5 * x,
    "sqrt": lambda x: 2.5 * np.sqrt(x),
    "log": lambda x: 2.5 * np.log1p(x),
}


class CountsOnBars(Scene):
    def construct(self):
        plot = make_counts_on_bars().build()
        self.play(Create(plot.panel))
        self.play(FadeIn(plot.layers["bars"], shift=UP * 0.2, lag_ratio=0.1))
        self.play(Write(plot.layers["counts"]), Write(plot.title))
        self.wait()


class LabelledScatter(Scene):
    def construct(self):
        plot = make_labelled_scatter().build()
        self.play(Create(plot.panel), Write(plot.title))
        self.play(FadeIn(plot.layers["points"], lag_ratio=0.05))
        self.play(Create(plot.layers["cutoff"]))
        self.play(FadeIn(plot.layers["best"], shift=DOWN * 0.1, lag_ratio=0.2))
        self.play(FadeIn(plot.layers["highlight"]), Write(plot.layers["note"]))
        self.wait()


class DirectLabels(Scene):
    def construct(self):
        plot = make_direct_labels().build()
        self.play(Create(plot.panel), Write(plot.title))
        self.play(
            Create(plot.layers["lines"].marks[0]),
            Create(plot.layers["lines"].marks[1]),
            Create(plot.layers["lines"].marks[2]),
        )
        self.play(Write(plot.layers["lines"].labels), FadeIn(plot.layers["note"]))
        self.wait()
