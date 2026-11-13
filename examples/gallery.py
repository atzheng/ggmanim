"""Phase 2 gallery: one scene per scales / guides / coords / themes feature.

Render a still of every scene (the last frame) into ``media/images/gallery/``::

    manim -ql -s -a examples/gallery.py

or one scene::

    manim -ql -s examples/gallery.py LogAxisMath

Each scene builds its plot(s) with a ``make_*`` function so the plots can also be
built (without rendering) from tests or a notebook.
"""

import warnings

import numpy as np
import pandas as pd
from manim import ORIGIN, RIGHT, UP, Create, FadeIn, Scene, VGroup, Write, config

from ggmanim import (
    aes,
    breaks_width,
    coord_cartesian,
    geom_col,
    geom_line,
    geom_point,
    geom_smooth,
    ggplot,
    guide_coloursteps,
    guide_legend,
    guides,
    label_comma,
    label_dollar,
    label_math,
    labs,
    scale_color_distiller,
    scale_color_fermenter,
    scale_color_viridis_c,
    scale_fill_brewer,
    scale_linetype_manual,
    scale_size,
    scale_x_continuous,
    scale_x_date,
    scale_x_log10,
    scale_y_continuous,
    scale_y_log10,
    theme,
    theme_bw,
    theme_dark,
    theme_gray,
    theme_void,
    xlim,
)

RNG_SEED = 7


def _rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(RNG_SEED + offset)


def build_small(plots, width: float, height: float) -> list:
    """Build several plots with a fixed panel size, for multi-plot scenes."""
    return [p.add(theme(panel_width=width, panel_height=height)).build() for p in plots]


def fit_frame(group: VGroup, margin: float = 0.3) -> VGroup:
    """Shrink ``group`` (never enlarge it) to fit the frame, and centre it."""
    w, h = config.frame_width - 2 * margin, config.frame_height - 2 * margin
    if group.width > w or group.height > h:
        group.scale(min(w / group.width, h / group.height))
    return group.move_to(ORIGIN)


# -- S1 + S9: categorical bar chart with rotated labels -------------------------------


def make_categorical_bars():
    df = pd.DataFrame(
        {
            "language": [
                "Python", "JavaScript", "TypeScript", "Rust", "Go", "Kotlin", "Haskell",
            ],
            "stars": [9400, 8100, 5200, 3900, 3100, 1500, 650],
            "paradigm": ["multi", "multi", "multi", "systems", "systems", "multi", "functional"],
        }
    )  # fmt: skip
    return (
        ggplot(df, aes("language", "stars", fill="paradigm"))
        .add(geom_col(width=0.7), name="bars")
        .add(
            scale_y_continuous(labels=label_comma(), expand=(0, 0, 0.05, 0)),
            labs(x=None, y="GitHub stars", fill="Paradigm", title="Discrete x, rotated labels"),
            theme(axis_text_x_angle=45),
        )
    )


class CategoricalBars(Scene):
    def construct(self):
        plot = make_categorical_bars().build()
        self.play(Create(plot.panel))
        self.play(FadeIn(plot.layers["bars"], shift=UP * 0.2, lag_ratio=0.1))
        self.play(Write(plot.legend), Write(plot.title))
        self.wait()


# -- S2: colourbar heat-style scatter -------------------------------------------------


def make_colourbar_scatter():
    rng = _rng(2)
    n = 400
    x = rng.uniform(-3, 3, n)
    y = rng.uniform(-3, 3, n)
    df = pd.DataFrame({"x": x, "y": y, "z": np.exp(-(x**2 + y**2) / 3) * np.cos(x * y)})
    return (
        ggplot(df, aes("x", "y", color="z"))
        .add(geom_point(size=0.07), name="points")
        .add(
            scale_color_viridis_c(option="magma"),
            labs(color="f(x, y)", title="Continuous colour, colourbar guide"),
        )
    )


class ColourbarScatter(Scene):
    def construct(self):
        plot = make_colourbar_scatter().build()
        self.add(plot.panel)
        self.play(FadeIn(plot.layers["points"], lag_ratio=0.002))
        self.play(FadeIn(plot.legend), Write(plot.title))
        self.wait()


# -- S3: size legend --------------------------------------------------------------------


def make_size_legend():
    df = pd.DataFrame(
        {
            "gdp": [1.2, 2.5, 4.1, 6.3, 9.8, 15.0, 22.0, 38.0, 45.0, 61.0],
            "life": [58, 63, 66, 70, 72, 74, 77, 80, 81, 83],
            "pop": [210, 35, 1400, 90, 330, 18, 125, 67, 9, 5],
        }
    )
    return (
        ggplot(df, aes("gdp", "life", size="pop"))
        .add(geom_point(alpha=0.6), name="points")
        .add(
            scale_x_log10(labels=label_dollar(suffix="k")),
            scale_size(range=(0.05, 0.35), breaks=[10, 100, 1000]),
            labs(x="GDP per capita", y="Life expectancy", size="Population (M)",
                 title="Mapped size, binned size legend"),
        )
    )  # fmt: skip


class SizeLegend(Scene):
    def construct(self):
        plot = make_size_legend().build()
        self.play(Create(plot.panel))
        self.play(FadeIn(plot.layers["points"], scale=0.5, lag_ratio=0.1))
        self.play(Write(plot.legend), Write(plot.title))
        self.wait()


# -- S4: shape + linetype legends --------------------------------------------------------


def make_shape_linetype():
    t = np.arange(1, 11)
    df = pd.DataFrame(
        {
            "epoch": np.tile(t, 3),
            "loss": np.concatenate([2.0 / t**0.5, 1.6 / t**0.7, 1.3 / t**0.4]) + 0.1,
            "model": np.repeat(["baseline", "wide", "deep"], len(t)),
        }
    )
    return (
        ggplot(df, aes("epoch", "loss", shape="model", linetype="model"))
        .add(geom_line(linewidth=3), name="lines")
        .add(geom_point(size=0.1), name="points")
        .add(
            scale_linetype_manual(values={"baseline": "solid", "wide": "dashed", "deep": "dotted"}),
            scale_x_continuous(breaks=breaks_width(2), minor_breaks=False),
            labs(x="Epoch", y="Validation loss", title="Shape + linetype, one merged legend"),
        )
    )


class ShapeLinetype(Scene):
    def construct(self):
        plot = make_shape_linetype().build()
        self.play(Create(plot.panel))
        self.play(Create(plot.layers["lines"]), FadeIn(plot.layers["points"]))
        self.play(Write(plot.legend), Write(plot.title))
        self.wait()


# -- S5: merged legend placed inside the panel ---------------------------------------------


def make_merged_inside_legend():
    rng = _rng(5)
    species = np.repeat(["setosa", "versicolor", "virginica"], 30)
    centre = {"setosa": (1.5, 0.25), "versicolor": (4.3, 1.3), "virginica": (5.5, 2.0)}
    pl = np.array([centre[s][0] for s in species]) + rng.normal(0, 0.35, 90)
    pw = np.array([centre[s][1] for s in species]) + rng.normal(0, 0.15, 90)
    df = pd.DataFrame({"petal_length": pl, "petal_width": pw, "species": species})
    return (
        ggplot(df, aes("petal_length", "petal_width", color="species", shape="species"))
        .add(geom_point(size=0.09, alpha=0.6), name="points")
        .add(
            guides(color=guide_legend(override_aes={"alpha": 1}, title="Species")),
            labs(x="Petal length", y="Petal width", shape="Species",
                 title="color + shape merged; legend inside the panel"),
            theme(legend_position=(0.02, 0.98), legend_background="#1c1c1c",
                  legend_border=True),
        )
    )  # fmt: skip


class MergedInsideLegend(Scene):
    def construct(self):
        plot = make_merged_inside_legend().build()
        self.play(Create(plot.panel))
        self.play(FadeIn(plot.layers["points"], lag_ratio=0.01))
        self.play(FadeIn(plot.legend), Write(plot.title))
        self.wait()


# -- S6: log axis with label_math and a minor grid --------------------------------------------


def make_log_axis():
    rng = _rng(6)
    n = 80
    mass = 10 ** rng.uniform(-1, 4, n)
    df = pd.DataFrame({"mass": mass, "rate": 3.4 * mass**0.75 * np.exp(rng.normal(0, 0.25, n))})
    return (
        ggplot(df, aes("mass", "rate"))
        .add(geom_point(alpha=0.8), name="points")
        .add(geom_smooth(method="lm"), name="fit")
        .add(
            scale_x_log10(labels=label_math()),
            scale_y_log10(labels=label_math()),
            labs(x="Body mass (kg)", y="Metabolic rate (W)",
                 title="log10 axes, label_math(), minor grid"),
            theme(grid_minor_opacity=0.15),
        )
    )  # fmt: skip


class LogAxisMath(Scene):
    def construct(self):
        plot = make_log_axis().build()
        self.play(Create(plot.panel))
        self.play(FadeIn(plot.layers["points"], lag_ratio=0.02))
        self.play(Create(plot.layers["fit"]), Write(plot.title))
        self.wait()


# -- S7: zoom (coord_cartesian) vs censor (scale limits) -------------------------------------


def _smooth_data() -> pd.DataFrame:
    rng = _rng(7)
    x = np.linspace(0, 10, 60)
    y = 0.15 * (x - 5) ** 2 + rng.normal(0, 0.8, len(x))
    y[(x > 8.5)] += 6  # an excursion at the right end the zoom hides
    return pd.DataFrame({"x": x, "y": y})


def make_zoom_vs_censor():
    base = (
        ggplot(_smooth_data(), aes("x", "y"))
        .add(geom_point(alpha=0.7), name="points")
        .add(geom_smooth(method="lm", degree=2), name="fit")
    )
    zoom = base.add(
        coord_cartesian(xlim=(0, 7)), labs(title="coord_cartesian(xlim=(0, 7))", subtitle="zoom")
    )
    censor = base.add(xlim(0, 7), labs(title="xlim(0, 7)", subtitle="censor, then refit"))
    return zoom, censor


class ZoomVsCensor(Scene):
    def construct(self):
        with warnings.catch_warnings():  # xlim() removing rows is the point here
            warnings.simplefilter("ignore", UserWarning)
            plots = build_small(make_zoom_vs_censor(), 4.6, 4.4)
        row = fit_frame(VGroup(*plots).arrange(RIGHT))
        self.play(FadeIn(row))
        self.wait()


# -- S8: monthly time series ----------------------------------------------------------------


def make_monthly_series():
    rng = _rng(8)
    months = pd.date_range("2022-01-01", periods=36, freq="MS")
    k = np.arange(len(months))
    sales = 120 + 1.8 * k + 18 * np.sin(2 * np.pi * k / 12) + rng.normal(0, 4, len(k))
    df = pd.DataFrame({"month": months, "sales": sales})
    return (
        ggplot(df, aes("month", "sales"))
        .add(geom_line(linewidth=3), name="line")
        .add(geom_point(size=0.05), name="points")
        .add(
            scale_x_date(date_breaks="6 months", date_labels="%b %Y",
                         date_minor_breaks="1 month"),
            labs(x=None, y="Sales (units)", title="Monthly time series: scale_x_date"),
        )
    )  # fmt: skip


class MonthlySeries(Scene):
    def construct(self):
        plot = make_monthly_series().build()
        self.play(Create(plot.panel))
        self.play(Create(plot.layers["line"]), FadeIn(plot.layers["points"]))
        self.play(Write(plot.title))
        self.wait()


# -- Themes: gray / bw / dark / void side by side --------------------------------------------


def make_theme_plots():
    rng = _rng(9)
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 1, 60),
            "y": rng.normal(0, 1, 60),
            "g": np.repeat(["a", "b", "c"], 20),
        }
    )
    base = ggplot(df, aes("x", "y", color="g")).add(geom_point(size=0.08), name="points")
    out = []
    for name, th in [
        ("theme_gray()", theme_gray()),
        ("theme_bw()", theme_bw()),
        ("theme_dark()", theme_dark()),
        ("theme_void()", theme_void()),
    ]:
        out.append(base.add(th, labs(title=name), theme(legend_position="none")))
    return out


class ThemeGallery(Scene):
    def construct(self):
        built = build_small(make_theme_plots(), 5.2, 2.2)
        grid = fit_frame(VGroup(*built).arrange_in_grid(rows=2, cols=2, buff=0.25))
        self.play(FadeIn(grid, lag_ratio=0.2))
        self.wait()


# -- ColorBrewer: brewer (discrete), distiller (continuous), fermenter (binned) --------------------


def make_brewer_plots():
    df = pd.DataFrame(
        {
            "region": ["North", "East", "South", "West", "Central"],
            "revenue": [7.4, 5.1, 4.8, 6.2, 3.5],
        }
    )
    brewer = (
        ggplot(df, aes("region", "revenue", fill="region"))
        .add(geom_col(width=0.7), name="bars")
        .add(
            scale_fill_brewer(palette="Set2"),
            labs(x=None, y="Revenue ($M)", fill="Region", title='scale_fill_brewer("Set2")'),
            theme(axis_text_x_angle=35),
        )
    )
    rng = _rng(10)
    x = rng.uniform(0, 1, 200)
    y = rng.uniform(0, 1, 200)
    d2 = pd.DataFrame({"x": x, "y": y, "z": np.sin(6 * x) + np.cos(4 * y)})
    distiller = (
        ggplot(d2, aes("x", "y", color="z"))
        .add(geom_point(size=0.07), name="points")
        .add(scale_color_distiller(palette="Spectral"),
             labs(title='scale_color_distiller("Spectral")'))
    )  # fmt: skip
    fermenter = (
        ggplot(d2, aes("x", "y", color="z"))
        .add(geom_point(size=0.07), name="points")
        .add(scale_color_fermenter(palette="YlGnBu", limits=(-2, 2),
                                   guide=guide_coloursteps(show_limits=True)),
             labs(title='scale_color_fermenter("YlGnBu")'))
    )  # fmt: skip
    return brewer, distiller, fermenter


class BrewerPalettes(Scene):
    def construct(self):
        row = fit_frame(VGroup(*build_small(make_brewer_plots(), 3.0, 3.6)).arrange(RIGHT))
        self.play(FadeIn(row))
        self.wait()


SCENES = [
    CategoricalBars,
    ColourbarScatter,
    SizeLegend,
    ShapeLinetype,
    MergedInsideLegend,
    LogAxisMath,
    ZoomVsCensor,
    MonthlySeries,
    ThemeGallery,
    BrewerPalettes,
]
