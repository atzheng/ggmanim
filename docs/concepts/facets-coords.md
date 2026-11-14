# Facets & coords

## Facets

`facet_wrap` splits the data into one panel per level of a variable and wraps the
panels into a grid; `facet_grid` lays them out as rows x columns of two sets of
variables. Scales are shared unless you free them, legends are shared, and the
built plot exposes every panel so you can animate them one at a time:

```python
import numpy as np
import pandas as pd
from manim import Create, FadeIn, Scene
from ggmanim import aes, facet_grid, facet_wrap, geom_hline, geom_point, ggplot, label_both

rng = np.random.default_rng(4)
cars = pd.DataFrame({"displ": rng.uniform(1.5, 6.5, 120),
                     "drv": rng.choice(["4wd", "front", "rear"], 120),
                     "cyl": rng.choice([4, 6, 8], 120)})
cars["hwy"] = 40 - 3 * cars["displ"] + rng.normal(0, 2, 120)

wrapped = (ggplot(cars, aes("displ", "hwy"))
           .add(geom_point(data=cars.drop(columns="drv"), colour="#777777", alpha=0.3),
                name="everyone")                 # no drv column: drawn in every panel
           .add(geom_point(aes(colour="drv")), name="points")
           .add(geom_hline(yintercept=cars["hwy"].mean()))  # also in every panel
           .add(facet_wrap("drv", ncol=3, scales="free_y", labeller=label_both))
           .build())

grid = (ggplot(cars, aes("displ", "hwy"))
        .geom_point()
        .add(facet_grid("drv ~ cyl", switch="y"))
        .build())

class RevealPanels(Scene):
    def construct(self):
        plot = wrapped
        for (row, col), panel in plot.panels.items():       # in panel order
            self.play(FadeIn(panel), FadeIn(plot.panel_grid.panel_strips[(row, col)]))
            self.play(Create(plot.layers["points"].panel(row, col)))
        self.add(plot)                                       # axis titles, other layers
```

- `facet_wrap(facets, nrow, ncol, scales, labeller, dir, as_table, drop,
  strip_position)`: `facets` is a column name, a list, `"~ a + b"`, or computed
  `{"name": lambda d: ...}`. With fixed scales, x tick labels go on the bottom
  panel of each column and y tick labels on the left of each row.
- `facet_grid(rows, cols, scales, space, margins, labeller, switch, drop)` (or
  `facet_grid("rows ~ cols")`): free x scales are per column and free y per row;
  `space="free_x"`/`"free_y"`/`"free"` sizes columns/rows by their data range;
  `margins=True` adds `"(all)"` panels; `switch` moves strips to the bottom/left.
- `scales="fixed"` / `"free_x"` / `"free_y"` / `"free"`; strip labellers
  `label_value`, `label_both`, `labeller(am={0: "auto"}, cyl=label_both)`
  (per variable; `default=` for the rest).
- A layer whose data lacks a facet variable is repeated in every panel (as in
  ggplot2); so are `geom_hline`/`geom_vline`/`geom_function`.
- The built plot: `plot.panels[(row, col)]`, `plot.strips` (all strips),
  `plot.panel_grid.panel_strips[(row, col)]`, `plot.layers[name].panel(row, col)`;
  `plot.panel`/`plot.axes` are the first panel. `theme(panel_width=,
  panel_height=)` size the whole grid; `panel_spacing`, `strip_*` style it.

`examples/facet_wrap.py` wraps seven panels into a 3 x 3 grid, with variants
for free scales, vertical layout and strips on the left:

```python
--8<-- "examples/facet_wrap.py"
```

`examples/facet_grid.py` and `examples/gallery_facets.py` show `facet_grid`
with free scales, `space="free"`, margins and switched strips.

## Coords

The coord turns the trained position scales into a panel. Geoms draw only
through the coord, so every geom works under every coord.

- `coord_cartesian(xlim=, ylim=, clip="on")`: the default. Its limits *zoom*:
  every row is kept and stats see the full data, unlike scale limits
  (`xlim()`, `scale_x_continuous(limits=)`), which *censor* rows before the stat.
- `coord_flip()` swaps the axes (horizontal bars); `coord_fixed(ratio=1)` fixes
  the aspect ratio.
- `coord_trans(x="log10", y="sqrt")` transforms at draw time: stats fit the
  untransformed data, and lines, polygons and rectangles are *munched* into
  curves. See `examples/coord_trans.py`.
- `coord_polar(theta="x", start=0, direction=1, inner_radius=0)`: bars become
  wedges. `geom_bar(width=1)` + `coord_polar(theta="y")` is a pie,
  `geom_col(width=1)` + `coord_polar()` a coxcomb. `coord_radar()` keeps the
  polar mapping but draws straight edges, for radar charts.

From `examples/polar.py`:

```python
import pandas as pd
from ggmanim import aes, coord_polar, coord_radar, geom_bar, geom_point, geom_polygon
from ggmanim import ggplot, labs, theme_void, ylim

FRUIT = pd.DataFrame({"fruit": ["apple"] * 6 + ["banana"] * 3 + ["cherry"] * 2 + ["kiwi"]})

pie = (ggplot(FRUIT, aes(x=1, fill="fruit"))
       .add(geom_bar(width=1, color="white", linewidth=2))
       .add(coord_polar(theta="y"), theme_void(), labs(title="pie: geom_bar + theta='y'"))
       .build())

RADAR = pd.DataFrame({
    "skill": pd.Categorical(["speed", "power", "range", "defence", "stamina"] * 2,
                            categories=["speed", "power", "range", "defence", "stamina"]),
    "player": ["A"] * 5 + ["B"] * 5,
    "score": [8, 5, 7, 4, 6, 4, 9, 5, 8, 7],
})

radar = (ggplot(RADAR, aes("skill", "score", color="player", group="player"))
         .add(geom_polygon(aes(fill="player"), alpha=0.2, linewidth=3), geom_point(size=0.08))
         .add(coord_radar(), ylim(0, 10), labs(title="radar: coord_radar()", x=None, y=None))
         .build())
```

Because a built plot is a mobject, the same data under two coords can be
morphed into each other; `BarsToPie` in `examples/polar.py` transforms a
stacked bar into a pie:

```python
from manim import Scene, Transform
from ggmanim import aes, coord_cartesian, coord_polar, geom_bar, ggplot, labs

class BarsToPie(Scene):
    def construct(self):
        bar = ggplot(FRUIT, aes(x=1, fill="fruit")).add(geom_bar(width=1), coord_cartesian())
        a = bar.add(labs(title="stacked bar")).build()
        b = bar.add(coord_polar(theta="y"), labs(title="coord_polar(theta='y')")).build()
        self.add(a)
        self.play(Transform(a, b), run_time=2)
```
