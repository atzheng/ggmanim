"""Data-keyed transitions between plots (``ggmanim.transition``).

Render (``-s`` gives only the last frame)::

    manim -ql examples/transitions.py ScatterYears     # 1952 -> 2007, keyed by country
    manim -ql examples/transitions.py BarsResort       # bars re-sort, y axis rescales
    manim -ql examples/transitions.py StatesLoop       # 4 states, wraps: GIF-friendly
    manim -ql examples/transitions.py RevealLines      # transition_reveal along x
    manim -ql --format=gif examples/transitions.py StatesLoop
"""

import numpy as np
import pandas as pd
from manim import Scene

from ggmanim import aes, geom_col, geom_line, geom_point, ggplot, labs, scale_x_log10
from ggmanim.transition import animate_reveal, animate_states, transition

CONTINENTS = {
    "Africa": ["Nigeria", "Egypt", "Kenya", "Ghana", "Ethiopia"],
    "Americas": ["Brazil", "Mexico", "Chile", "Canada", "Peru"],
    "Asia": ["China", "India", "Japan", "Vietnam", "Korea"],
    "Europe": ["France", "Poland", "Spain", "Sweden", "Italy"],
}
YEARS = (1952, 1972, 1992, 2007)


def countries() -> pd.DataFrame:
    """A small synthetic gapminder: GDP per capita and life expectancy."""
    rng = np.random.default_rng(7)
    rows = []
    base = {"Africa": (900, 40), "Americas": (3000, 52), "Asia": (1000, 45), "Europe": (5000, 64)}
    for cont, names in CONTINENTS.items():
        g0, l0 = base[cont]
        for name in names:
            g = g0 * rng.lognormal(0, 0.4)
            life = l0 + rng.normal(0, 4)
            growth = rng.uniform(0.01, 0.045)
            for year in YEARS:
                t = year - 1952
                rows.append(
                    {
                        "country": name,
                        "continent": cont,
                        "year": year,
                        "gdp": g * (1 + growth) ** t,
                        "life": min(life + t * rng.uniform(0.2, 0.45), 84),
                    }
                )
    df = pd.DataFrame(rows)
    # a few countries only appear later / disappear (they fade in / out)
    drop = (df.country.isin(["Vietnam", "Korea"]) & (df.year < 1972)) | (
        (df.country == "Ghana") & (df.year > 1992)
    )
    return df[~drop].reset_index(drop=True)


def scatter(df: pd.DataFrame, year: int):
    return (
        ggplot(df[df.year == year], aes("gdp", "life", colour="continent"))
        .add(geom_point(size=0.09), name="points")
        .add(scale_x_log10())
        .add(labs(title=f"Year {year}", x="GDP per capita", y="Life expectancy"))
    )


def bars(df: pd.DataFrame, year: int):
    d = df[(df.year == year) & (df.continent == "Europe")].sort_values("gdp")
    # x in GDP order (re-sorts between years); fill by name keeps each bar's colour
    d = d.assign(name=d.country, country=pd.Categorical(d.country, categories=list(d.country)))
    return (
        ggplot(d, aes("country", "gdp", fill="name"))
        .add(geom_col(show_legend=False), name="bars")
        .add(labs(title=f"Europe, {year}", x=None, y="GDP per capita"))
    )


class ScatterYears(Scene):
    def construct(self):
        df = countries()
        t = transition(scatter(df, 1952), scatter(df, 2007), key="country", run_time=2.5)
        self.add(t.plot_a)
        self.wait(0.5)
        self.play(t)
        self.wait(0.5)


class BarsResort(Scene):
    def construct(self):
        df = countries()
        t = transition(bars(df, 1952), bars(df, 2007), key="country", run_time=2)
        self.add(t.plot_a)
        self.wait(0.5)
        self.play(t)
        self.wait(0.5)


class StatesLoop(Scene):
    """gganimate's ``transition_states``: 4 years, wrapping (a looping GIF)."""

    def construct(self):
        df = countries()
        spec = (
            ggplot(df, aes("gdp", "life", colour="continent"))
            .add(geom_point(size=0.09), name="points")
            .add(scale_x_log10())
            .add(labs(x="GDP per capita", y="Life expectancy"))
        )
        anim = animate_states(
            spec,
            "year",
            key="country",
            title="Year {state}",
            transition_length=1.0,
            state_length=0.4,
            wrap=True,
        )
        self.play(anim)


class RevealLines(Scene):
    """gganimate's ``transition_reveal``: lines grow along the x axis."""

    def construct(self):
        x = np.linspace(0, 10, 60)
        df = pd.concat(
            [
                pd.DataFrame({"x": x, "y": np.sin(x + p) + 0.1 * x, "series": s})
                for s, p in (("a", 0.0), ("b", 1.5), ("c", 3.0))
            ]
        )
        spec = ggplot(df, aes("x", "y", colour="series")).add(geom_line(), name="lines")
        self.play(animate_reveal(spec, steps=12, run_time=3))
        self.wait(0.3)
