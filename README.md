# ggmanim

A grammar of graphics for [manim](https://www.manim.community/): ggplot2-style
plots that build into manim mobjects you can animate piece by piece.

```text
ggplot2                                   ggmanim
ggplot(df, aes(x, y, colour = g)) +       ggplot(df, aes("x", "y", color="g"))
  geom_point(alpha = 0.7) +                 .add(geom_point(alpha=0.7))
  geom_smooth(method = "lm") +              .add(geom_smooth(method="lm"))
  scale_x_log10() +                         .add(scale_x_log10())
  labs(x = "Dose")                          .add(labs(x="Dose"))
```

Differences from ggplot2 worth knowing up front:

- **No `+`.** Components are combined with `.add(...)`, which returns a *new* plot
  (plots are immutable; the DataFrame is shared, not copied). Every registered
  component also has a chaining method: `p.geom_point(alpha=0.7)` is
  `p.add(geom_point(alpha=0.7))`.
- **Column names are strings**: `aes("x", "y", color="group")`. A callable
  (`aes(y=lambda d: d.a / d.b)`) computes a column; `after_stat("level")` refers to
  a variable computed by the stat.
- **`.build()` returns a manim mobject** (`GGPlot`, a `VGroup`) with named parts:
  `plot.panel`, `plot.axes`, `plot.layers[name]`, `plot.legend`, `plot.title`.
  Building never mutates the spec, so you can build the same plot twice.
- **Sizes are in manim scene units** (`geom_point(size=0.08)` is the dot radius).
- **Text uses Pango (`Text`) by default**, so no LaTeX installation is needed.
  LaTeX is opt-in with `theme(text_engine="tex")` (see below).
- Foreground colours follow manim's background: white axes and text on the
  default black background, black on a white one.

## Install

```bash
pip install git+https://github.com/atzheng/ggmanim                    # manim, pandas, numpy
pip install "ggmanim[stats] @ git+https://github.com/atzheng/ggmanim"  # + scipy
```

Requires Python 3.10+. The `stats` extra (`scipy`) is optional: it gives exact
t-quantiles for `geom_smooth` confidence bands.

## Example 1: scatter, linear fits, log x axis

```python
import numpy as np
import pandas as pd
from ggmanim import aes, geom_point, geom_smooth, ggplot, labs, scale_x_log10

rng = np.random.default_rng(1)
dose = 10 ** rng.uniform(0, 3, size=60)
df = pd.DataFrame({
    "dose": dose,
    "compound": np.repeat(["A", "B", "C"], 20),
    "response": 1 + np.repeat([1.6, 1.0, 0.4], 20) * np.log10(dose) + rng.normal(0, 0.3, 60),
})

p = (ggplot(df, aes("dose", "response", color="compound"))
     .add(geom_point(alpha=0.8), name="points")
     .add(geom_smooth(method="lm"), name="fit")
     .add(scale_x_log10(), labs(x="Dose (mg)", y="Response", color="Compound")))

plot = p.build()            # a GGPlot mobject, centred in the frame
points = plot.layers["points"]
```

Inside a scene, animate the parts in whatever order you like:

```python
from manim import Create, FadeIn, Scene, Write

class DoseResponse(Scene):
    def construct(self):
        plot = p.build()
        self.play(Create(plot.panel))                    # grid, axes, ticks, titles
        self.play(FadeIn(plot.layers["points"], lag_ratio=0.02))
        self.play(Create(plot.layers["fit"]))
        self.play(Write(plot.legend))
```

`examples/basic.py` is this plot as a runnable scene:
`manim -ql examples/basic.py Basic`.

## Example 2: grouped lines, reference lines, a theme

```python
import numpy as np
import pandas as pd
from ggmanim import aes, ggplot, theme_minimal

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
        .add(theme_minimal())
        .guides(color="none")                            # the line labels replace the legend
        .labs(x="time", y=None, title="Two waves")
        .build())
```

`geom_vline`/`geom_hline`/`geom_function` layers take part in the automatic axis
limits, so reference lines are never off-panel.

## Example 3: bars and a function, with explicit scales

```python
import numpy as np
import pandas as pd
from ggmanim import aes, geom_col, geom_function, ggplot, scale_fill_manual, theme, ylim
from manim import BLUE, ORANGE, RED

df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [3.0, 5.0, 4.0, 6.5, 6.0],
                   "kind": ["old", "old", "new", "new", "new"]})

p = (ggplot(df, aes("x", "y", fill="kind"))
     .add(geom_col(width=0.6),
          geom_function(lambda x: 2 + 0.9 * x, color=RED),
          scale_fill_manual(values={"old": BLUE, "new": ORANGE}),
          ylim(0, 8),
          theme(legend_position="bottom")))
plot = p.build()
```

### LaTeX labels (opt-in)

Labels, tick labels and legend text are Pango `Text` by default. To typeset them
with LaTeX instead (requires a LaTeX installation, as for manim's `Tex`):

```python
from ggmanim import ggplot, theme

tex_plot = ggplot().geom_function(lambda x: x ** 2, xlim=(0, 2)).add(
    theme(text_engine="tex")).labs(x=r"$x$", y=r"$x^2$")
```

Any label may also be a ready-made manim `Mobject`, e.g. `labs(y=MathTex(r"\hat{y}"))`.

## Documentation

The full documentation is at **<https://atzheng.github.io/ggmanim/>** (sources in
[`docs/`](docs/)):

- [Getting started](docs/getting-started.md) and [Concepts](docs/concepts/index.md):
  immutable specs, the build pipeline, scene units, Text vs Tex, foreground colours.
- One guide per part of the grammar: [aesthetics, layers and the rest](docs/concepts/grammar.md),
  [scales & guides](docs/concepts/scales-guides.md),
  [geoms, stats & positions](docs/concepts/geoms.md),
  [facets & coords](docs/concepts/facets-coords.md), [themes](docs/concepts/themes.md).
- [Animation](docs/concepts/animation.md): `plot.create()`, per-geom animations,
  mark keys, data-keyed transitions between plots.
- [Performance](docs/concepts/performance.md): batched marks for large layers.
- [The full ggplot2 → ggmanim cheat sheet](docs/cheatsheet.md) and the API reference.

Build it locally with `pip install -e '.[docs]' && mkdocs serve`.

## Gallery

[`examples/gallery.py`](examples/gallery.py) has one scene per feature: a
categorical bar chart with rotated labels, a colourbar scatter, a size legend,
shape + linetype legends, a merged legend placed inside the panel, log axes with
`label_math()` and a minor grid, zoom vs censoring side by side, a monthly time
series, `theme_gray`/`bw`/`dark`/`void` side by side, and ColorBrewer palettes
(brewer, distiller, and fermenter with `guide_coloursteps`).
Render stills of all of them with

```bash
manim -ql -s -a examples/gallery.py
```

[`examples/gallery_facets.py`](examples/gallery_facets.py) shows facets:
`facet_wrap` with fixed and free y scales, `facet_grid` with switched strips and
with `space="free_y"`, a non-faceted reference layer repeated in every panel, and
an animation revealing the panels one by one (`manim -ql -s -a
examples/gallery_facets.py`).

[`examples/gallery_geoms.py`](examples/gallery_geoms.py) has one scene per
geom/stat/position idea: stacked and dodged bars with their counts written on them, a
histogram with density curves, violin + boxplot + jitter, a step chart with
error bars, a stacked area, a tile heatmap, an annotated scatter
(text/label/segment/rect), loess vs lm, 2-D density contours, and nine of them on
one frame (`GeomsOverview`). It reuses the plots and data of the per-feature examples
(`histograms.py`, `distributions.py`, `areas_heatmap.py`, `annotations.py`,
`smoothers.py`, `two_d.py`), which have more scenes each, as do `positions.py`
and `intervals.py`.

CI renders every `examples/*.py` (`scripts/render_examples.sh`) on every push and
uploads the PNGs as the `examples` artifact.

## ggplot2 → ggmanim in brief

Names and arguments follow ggplot2 unless noted; the
[full cheat sheet](docs/cheatsheet.md) lists every geom, stat, position, scale,
guide, facet, coord and theme, and what is not there yet.

| ggplot2 | ggmanim |
|---|---|
| `ggplot(df, aes(x, y, colour = g)) + geom_point()` | `ggplot(df, aes("x", "y", color="g")).add(geom_point())` or `.geom_point()` |
| `geom_bar()`, `geom_histogram()`, `geom_smooth()`, `geom_boxplot()`, ... | same, with the same stats and positions |
| `aes(y = after_stat(density))` | `aes(y=after_stat("density"))` |
| `scale_x_log10()`, `scale_colour_viridis_c()`, `scales::label_percent()` | same (`color`/`colour`/`fill` variants) |
| `facet_wrap(~class)`, `facet_grid(drv ~ cyl)` | `facet_wrap("class")`, `facet_grid("drv ~ cyl")` |
| `coord_flip()`, `coord_polar(theta = "y")`, `coord_trans(x = "log10")` | same |
| `theme_bw() + theme(legend.position = "bottom")` | `.add(theme_bw(), theme(legend_position="bottom"))` |
| `labs()`, `guides(colour = "none")` | same |
| gganimate `transition_states(year)` | `animate_states(spec, "year", key="country")` |
| (manim only) | `plot.create()`, `plot.reveal_layer("fit")`, `plot.mark("points", "Ohio")`, `transition(p1, p2, key=...)` |

## Development

```bash
pip install -e '.[dev]'
ruff check . && ruff format --check .
mypy
pytest
manim -ql -s examples/basic.py Basic
mkdocs build --strict      # docs; needs the docs extra
```
