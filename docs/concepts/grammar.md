# The grammar

A plot is data plus a stack of independent components. This page goes through
them in the order the [build pipeline](index.md#the-build-pipeline) uses them;
the pages that follow go into more depth. All snippets run as they are (with
`import numpy as np`, `import pandas as pd` and the `ggmanim` imports shown).

The running example is `examples/basic.py`:

```python
--8<-- "examples/basic.py"
```

## Aesthetics

`aes()` maps visual properties to data. Values are column names (strings),
callables that compute a column, or `after_stat(...)` references to variables
computed by the stat:

```python
from ggmanim import aes, after_stat

aes("dose", "response", color="compound")          # columns
aes(x="t", y=lambda d: d.value / d.value.max())     # a computed column
aes("x", fill=after_stat("count"))                  # the stat's output
aes("gdp", "life", key="country")                   # identifies marks for animation
```

The plot's mapping is inherited by every layer (`inherit_aes=True`); a layer's
own mapping is merged over it, and `aes(color=None)` removes an inherited
aesthetic. `key` is ggmanim's addition: it is never scaled and never groups,
and ends up in `layer.keys` so that animations can address single marks (see
[Animation](animation.md)).

## Layers

A layer is a geom + a stat + a position + data + mapping + parameters. Each
`geom_*` or `stat_*` call makes one layer; `.add(..., name=...)` names it:

```python
import numpy as np
import pandas as pd
from ggmanim import aes, ggplot

t = np.linspace(0, 10, 50)
df = pd.DataFrame({
    "t": np.tile(t, 2),
    "value": np.concatenate([np.sin(t), 0.5 * np.cos(t)]),
    "series": np.repeat(["sin", "cos"], len(t)),
})

plot = (ggplot(df, aes("t", "value", color="series"))
        .geom_line(label=["cos", "sin"], name="lines")   # one label per group, at its end
        .geom_hline(yintercept=0, alpha=0.5)
        .geom_vline(xintercept=[np.pi, 2 * np.pi], linewidth=1)
        .guides(color="none")                            # the line labels replace the legend
        .labs(x="time", y=None, title="Two waves")
        .build())
```

A layer may bring its own `data=`; layers without a facet variable are
repeated in every facet panel.

## Geoms

Geoms draw. Each one turns the computed layer data into manim mobjects through
the coord, and attaches `.marks`/`.keys` so every mark can be animated. There
are geoms for points, lines and paths, bars and columns, areas and ribbons,
rectangles and tiles, text and labels, segments and curves, reference lines,
intervals, boxplots, violins, densities, contours, hexagons and more; see
[Geoms, stats & positions](geoms.md) and the [geom reference](../api/geoms.md).

## Stats

Stats transform the data before it is drawn: `geom_histogram` bins
(`stat_bin`), `geom_bar` counts (`stat_count`), `geom_smooth` fits
(`stat_smooth`), `geom_boxplot` summarises (`stat_boxplot`). Computed variables
are mapped with `after_stat`:

```python
import numpy as np
import pandas as pd
from ggmanim import aes, after_stat, geom_density, geom_histogram, ggplot

rng = np.random.default_rng(0)
df = pd.DataFrame({"x": np.concatenate([rng.normal(0, 1, 300), rng.normal(3, 0.6, 150)])})

hist = (ggplot(df, aes("x"))
        .add(geom_histogram(aes(y=after_stat("density")), binwidth=0.25, alpha=0.6))
        .add(geom_density(adjust=0.8))
        .build())
```

## Positions

Positions resolve overlaps after the stat: bars stack by default and can be
filled, dodged or nudged; points can be jittered. A position is a name
(`position="dodge"`), a name with arguments (`position="dodge(width=0.8)"`) or
an object (`position=position_dodge(preserve="single")`). See
[Positions](geoms.md#positions).

## Scales

Every mapped aesthetic is trained and mapped by a scale. Without an explicit
scale ggmanim picks one from the data; adding one changes its breaks, labels,
limits, transform or palette:

```python
import pandas as pd
from ggmanim import (aes, breaks_width, ggplot, label_percent, scale_fill_brewer,
                     scale_x_discrete, scale_y_continuous, theme)

cities = pd.DataFrame({"city": ["Lima", "Oslo", "Pune", "Kyiv"],
                       "share": [0.42, 0.17, 0.30, 0.11]})

bars = (ggplot(cities, aes("city", "share", fill="city"))
        .geom_col(width=0.6)
        .add(scale_x_discrete(limits=["Oslo", "Kyiv", "Pune", "Lima"]),
             scale_y_continuous(labels=label_percent(), breaks=breaks_width(0.1)),
             scale_fill_brewer(palette="Set2"),
             theme(axis_text_x_angle=45, legend_position="none"))
        .build())
```

Non-position scales get a guide (legend, colourbar or colour steps). See
[Scales & guides](scales-guides.md).

## Coords

The coord turns trained position scales into a panel: `coord_cartesian`
(zoom with `xlim=`/`ylim=` without dropping rows), `coord_flip`,
`coord_fixed`, `coord_trans` and `coord_polar`/`coord_radar`. See
[Facets & coords](facets-coords.md#coords).

## Facets

Facets split the data into panels: `facet_wrap("var")` wraps one panel per
level into a grid, `facet_grid("rows ~ cols")` crosses two sets of variables.
See [Facets & coords](facets-coords.md).

## Themes

The theme controls everything that is not data: text sizes and fonts, axes,
grids, backgrounds, legend placement, panel size. Complete themes
(`theme_bw()`, `theme_minimal()`, ...) replace the current theme; `theme(...)`
overrides single elements on top of it. See [Themes](themes.md).
