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

## Scales

Every mapped aesthetic is trained and mapped by a scale. Without an explicit
scale, ggmanim picks one from the data, as ggplot2 does:

| data | `x` / `y` | `color` / `fill` | `size` / `alpha` / `linewidth` | `shape` / `linetype` |
|---|---|---|---|---|
| numeric | continuous | gradient + colourbar | range, binned legend | error (use a discrete column) |
| string / categorical / bool | discrete (`1..n`) | hue palette + legend | evenly spaced range | palette + legend |
| dates / datetimes | `scale_*_date` / `_datetime` | — | — | — |

Add a scale to change its breaks, labels, limits, transform or palette:

```python
import numpy as np
import pandas as pd
from ggmanim import (aes, breaks_width, coord_flip, ggplot, label_comma, label_math,
                     label_percent, scale_color_viridis_c, scale_fill_brewer,
                     scale_shape_manual, scale_size, scale_x_date, scale_x_discrete,
                     scale_x_log10, scale_y_continuous, theme)

rng = np.random.default_rng(3)
cities = pd.DataFrame({"city": ["Lima", "Oslo", "Pune", "Kyiv"],
                       "share": [0.42, 0.17, 0.30, 0.11],
                       "pop": [10.7, 0.7, 7.4, 3.0]})

# discrete x (order set by limits), percent labels, rotated tick labels, brewer fill
bars = (ggplot(cities, aes("city", "share", fill="city"))
        .geom_col(width=0.6)
        .add(scale_x_discrete(limits=["Oslo", "Kyiv", "Pune", "Lima"]),
             scale_y_continuous(labels=label_percent(), breaks=breaks_width(0.1)),
             scale_fill_brewer(palette="Set2"),
             theme(axis_text_x_angle=45, legend_position="none"))
        .build())

# log axis with 10^k labels and a minor grid; continuous colour; size legend
pts = pd.DataFrame({"mass": 10 ** rng.uniform(-1, 4, 50)})
pts["rate"] = 3.4 * pts["mass"] ** 0.75
pts["temp"] = rng.uniform(0, 40, 50)
log_plot = (ggplot(pts, aes("mass", "rate", color="temp", size="mass"))
            .geom_point()
            .add(scale_x_log10(labels=label_math()),
                 scale_y_continuous(labels=label_comma()),
                 scale_color_viridis_c(option="plasma"),
                 scale_size(range=(0.03, 0.2), breaks=[1, 100, 10000]),
                 theme(grid_minor_opacity=0.15))
            .build())

# dates on x: monthly breaks with strftime labels; a horizontal bar chart via coord_flip
months = pd.DataFrame({"month": pd.date_range("2024-01-01", periods=12, freq="MS"),
                       "sales": rng.normal(100, 10, 12)})
ts = (ggplot(months, aes("month", "sales")).geom_line()
      .add(scale_x_date(date_breaks="3 months", date_labels="%b %Y")).build())
hbars = ggplot(cities, aes("city", "pop")).geom_col().add(coord_flip()).build()

shapes = (ggplot(cities, aes("pop", "share", shape="city")).geom_point(size=0.1)
          .add(scale_shape_manual(values=["circle", "triangle_open", "star", "cross"]))
          .build())
```

- **Position**: `scale_x_continuous` / `scale_y_continuous` (`breaks`, `labels`,
  `limits`, `expand`, `oob`, `trans`, `n_breaks`, `minor_breaks`, `position="top"`
  / `"right"`), `scale_*_log10` / `_sqrt` / `_reverse`, `xlim` / `ylim`,
  `scale_*_discrete` (`limits` sets the order and subset of levels),
  `scale_*_date` / `scale_*_datetime` (`date_breaks="1 month"`, `date_labels="%b %Y"`,
  `date_minor_breaks`; limits and breaks may be dates or ISO strings).
- **Colour / fill** (`color`, `colour` and `fill` variants of each):
  discrete `_discrete` / `_hue`, `_manual`, `_identity`, `_viridis_d`, `_brewer`;
  continuous `_continuous`, `_gradient`, `_gradient2` (diverging around `midpoint`),
  `_gradientn`, `_viridis_c`, `_distiller`; binned `_fermenter` (guide:
  `guide_coloursteps`).
- **Size / alpha / linewidth**: `scale_size` (area-proportional), `scale_radius`,
  `scale_size_area` (0 maps to 0), `scale_alpha`, `scale_linewidth`, each with
  `range=` and an `_identity` variant. Sizes are scene units, linewidths manim
  stroke widths.
- **Shape / linetype**: `scale_shape(solid=...)`, `scale_shape_manual`,
  `scale_linetype`, `scale_linetype_manual`, `_identity`. Shapes: `circle`,
  `triangle`, `square`, `diamond`, `star` (+ `_open`), `cross`, `plus`, or ggplot2's
  `pch` codes; linetypes: `solid`, `dashed`, `dotted`, `dotdash`, `longdash`,
  `twodash`, `blank` or a hex pattern such as `"44"`.
- **Labels and breaks** (`ggmanim.labels`, after the `scales` package): `label_number`,
  `label_comma`, `label_percent`, `label_dollar` / `label_currency`,
  `label_scientific`, `label_math` (`10^k`, Unicode superscripts with the Text
  engine, LaTeX with `text_engine="tex"`), `label_date`, `label_wrap`;
  `breaks_pretty`, `breaks_extended`, `breaks_width`, `breaks_log`.

**Limits vs zoom.** As in ggplot2, scale limits (`xlim(0, 7)`,
`scale_x_continuous(limits=...)`) *censor*: rows outside are dropped (with a
warning) before stats run, so a smooth is refitted on what is left.
`coord_cartesian(xlim=(0, 7))` *zooms*: every row is kept and the stat sees the full
data; marks are clipped at the panel edge (`clip="off"` to let them overflow).
`coord_fixed(ratio=1)` fixes the aspect ratio; `coord_flip()` swaps the axes.

## Guides

Every mapped non-position aesthetic gets a guide: a legend of keys for discrete
scales and for size/alpha/linewidth (binned at the scale's breaks), a colourbar
for continuous colour, and colour steps (`guide_coloursteps`, one step per bin,
labels at the bin boundaries) for binned colour scales such as `scale_*_fermenter`. As in
ggplot2, legends with the same title and the same levels merge into one with
composite keys, so `color="g", shape="g"` gives a single legend. Configure a
guide with `guide_legend(...)` / `guide_colorbar(...)`, either in `guides()` or
as a scale's `guide=`; place and style them with `theme()`:

```python
import numpy as np
import pandas as pd
from ggmanim import (aes, ggplot, guide_colorbar, guide_coloursteps, guide_legend, guides,
                     scale_color_fermenter, theme)

df = pd.DataFrame({"x": np.arange(12.0), "y": np.sqrt(np.arange(12.0)),
                   "g": list("abcd") * 3, "z": np.linspace(0, 1, 12)})

legend_plot = (ggplot(df, aes("x", "y", color="g", shape="g"))   # one merged legend
               .geom_point(alpha=0.5)
               .add(guides(color=guide_legend(ncol=2, override_aes={"alpha": 1})),
                    theme(legend_position=(1, 0),                # inside, bottom-right
                          legend_background="#222222", legend_border=True))
               .build())

two_guides = (ggplot(df, aes("x", "y", color="z", shape="g"))
              .geom_point()
              .add(guides(shape=guide_legend(order=1), color=guide_colorbar(order=2)),
                   theme(legend_position="bottom", legend_justification="left"))
              .build())

steps = (ggplot(df, aes("x", "y", color="z"))                   # binned colour scale
         .geom_point()
         .add(scale_color_fermenter(palette="YlGnBu", limits=(0, 1),
                                    guide=guide_coloursteps(show_limits=True)))
         .build())
```

- `guide_legend(title, nrow, ncol, byrow, reverse, override_aes, order, direction)`;
  `guide_colorbar(title, barwidth, barheight, nbin, reverse, ticks, order, direction)`;
  `guide_coloursteps(title, barwidth, barheight, even_steps, show_limits, reverse,
  ticks, order, direction)` (alias `guide_colorsteps`), e.g.
  `scale_fill_fermenter(limits=(0, 10), guide=guide_coloursteps(show_limits=True))`.
- `theme(legend_position=...)`: `"right"` (default), `"left"`, `"top"`, `"bottom"`,
  `"none"`, or `(x, y)` in `[0, 1]` to put the legend *inside* the panel;
  `legend_justification` picks the point of the legend box anchored there.
- `legend_direction` (key layout, independent of position), `legend_box`
  (stack several legends `"vertical"`ly or `"horizontal"`ly), `legend_key_size`,
  `legend_key_width`, `legend_spacing`, `legend_box_spacing`, `legend_title_align`,
  `legend_background`, `legend_border`, `legend_text_color`/`_font`,
  `legend_title_color`/`_font`.
- `show_legend=False` on a layer keeps it out of every legend;
  `show_legend={"color": False}` only out of the colour legend.

## Themes

`theme_default()` (= `theme_manim()`) is drawn straight on the scene background,
with foreground colours that follow it. The ggplot2 looks `theme_gray()` (alias
`theme_grey()`), `theme_bw()`, `theme_linedraw()`, `theme_light()` and
`theme_dark()` bring their own white plot background; `theme_minimal()`,
`theme_classic()` and `theme_void()` are background-agnostic. Complete themes
replace the current theme; `theme(...)` overrides single elements on top of it:

```python
from ggmanim import theme, theme_bw

themed = (ggplot(cities, aes("city", "share"))
          .geom_col()
          .add(theme_bw(), theme(axis_text_x_angle=30, panel_grid_minor=False,
                                 panel_background="#F4F1EA", plot_title_position="plot"))
          .labs(title="Share by city")
          .build())
```

Elements include text sizes and fonts, per-axis switches (`axis_text_x=False`,
`axis_ticks_y=False`, ...), tick-label rotation and justification
(`axis_text_x_angle`, `axis_text_x_hjust`), `panel_background`, `panel_border`,
major/minor grid per direction, `plot_background`, `plot_margin`,
`panel_width`/`panel_height` and all the `legend_*` elements above. See the
`Theme` docstring for the full list.

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

CI renders every `examples/*.py` (`scripts/render_examples.sh`) on every push and
uploads the PNGs as the `examples` artifact.

## ggplot2 → ggmanim cheat sheet

Everything below is supported; names and arguments follow ggplot2 unless noted.
Each section ends with what is not there yet; see [PLAN.md](PLAN.md) for the
roadmap.

| ggplot2 | ggmanim | notes |
|---|---|---|
| `ggplot(df, aes(x, y, colour = g)) + ...` | `ggplot(df, aes("x", "y", color="g")).add(...)` | strings for columns; `colour` accepted |
| `labs()`, `xlab()`, `ylab()`, `ggtitle()` | `labs()`, `xlab()`, `ylab()`, `labs(title=...)` | |
| `p1 + p2` (patchwork) | build both, arrange the mobjects | see `examples/gallery.py` |

<!-- section: geoms -->
### Geoms

| ggplot2 | ggmanim | notes |
|---|---|---|
| `geom_point()`, `geom_line()`, `geom_path()` | same | `size` = radius in scene units |
| `geom_col()`, `geom_bar(stat = "identity")` | `geom_col()`, `geom_bar()` | identity heights only |
| `geom_smooth(method = "lm")` | `geom_smooth(method="lm", degree=1)` | polynomial fits, SE ribbon |
| `geom_function(fun = f)` | `geom_function(f)` | |
| `geom_hline()`, `geom_vline()` | same | take part in axis limits |

Not yet: `geom_histogram`, `geom_area`/`ribbon`, `geom_text`/`label`, `geom_boxplot`,
`geom_violin`, `geom_density`, `geom_rect`/`tile`, `geom_segment`, error bars.

<!-- section: stats -->
### Stats

| ggplot2 | ggmanim | notes |
|---|---|---|
| `stat_identity()`, `stat_smooth()`, `stat_function()` | same | `after_stat("var")` maps computed variables |

Not yet: `stat_count`, `stat_bin`, `stat_summary`, `stat_density`, `stat_boxplot`.

<!-- section: positions -->
### Positions

| ggplot2 | ggmanim | notes |
|---|---|---|
| `position = "identity"`, `position_identity()` | same | the default for most geoms |
| `position_stack(vjust, reverse)`, `position_fill()` | same | default for `geom_col`/`geom_bar`; first group on top, negatives stacked below 0 |
| `position_dodge(width, preserve)` | same | `preserve="single"` keeps equal bar widths |
| `position_dodge2(width, preserve, padding)` | same | dodges by interval overlap (variable widths) |
| `position_jitter(width, height, seed)`, `geom_jitter()` | same | numpy `default_rng(seed)`; `seed=None` is derived from the data, so builds are reproducible |
| `position_jitterdodge(jitter.width, jitter.height, dodge.width)` | `position_jitterdodge(jitter_width, jitter_height, dodge_width)` | needs a mapped fill/colour/... |
| `position_nudge(x, y)` | same | offsets in transformed units |

Positions accept a name (`position="dodge"`), a class or an instance
(`position=position_dodge(width=0.8)`). See `examples/positions.py`.

<!-- section: scales -->
### Scales

| ggplot2 | ggmanim | notes |
|---|---|---|
| `scale_x_continuous()`, `scale_x_log10()`, `scale_x_sqrt()`, `scale_x_reverse()` | same (`x` and `y`) | `position="top"`/`"right"` |
| `xlim()`, `ylim()` | `xlim(lo, hi)`, `ylim(lo, hi)` | censor out-of-range rows |
| `scale_x_discrete()` | same (`x` and `y`) | |
| `scale_x_date()`, `scale_x_datetime()` | same (`x` and `y`) | `date_breaks`, `date_labels`, `date_minor_breaks` |
| `scale_colour_manual()`, `_identity()`, `_discrete()`, `_hue()` | same (`color`/`colour`/`fill`) | |
| `scale_colour_gradient()`, `_gradient2()`, `_gradientn()`, `_continuous()` | same | Oklab interpolation |
| `scale_colour_viridis_c()`, `_viridis_d()` | same | viridis, magma, inferno, plasma, cividis |
| `scale_colour_brewer()`, `_distiller()`, `_fermenter()` | same | all ColorBrewer palettes |
| `scale_size()`, `scale_radius()`, `scale_size_area()` | same | radius in scene units |
| `scale_alpha()`, `scale_linewidth()` | same | |
| `scale_shape()`, `scale_shape_manual()` | same | names or `pch` codes |
| `scale_linetype()`, `scale_linetype_manual()` | same | names, codes or hex patterns |
| `scales::label_percent()` etc. | `label_percent()`, `label_comma()`, `label_dollar()`, `label_number()`, `label_scientific()`, `label_math()`, `label_date()`, `label_wrap()` | |
| `scales::breaks_width()` etc. | `breaks_width()`, `breaks_pretty()`, `breaks_extended()`, `breaks_log()` | |

Not yet: binned position scales, `sec_axis`.

<!-- section: guides -->
### Guides

| ggplot2 | ggmanim | notes |
|---|---|---|
| `guides(colour = "none")` | `guides(color="none")` | |
| `guide_legend()` | `guide_legend(title, nrow, ncol, byrow, reverse, override_aes, order, direction)` | |
| `guide_colourbar()` | `guide_colorbar(...)` / `guide_colourbar(...)` | `barwidth`, `barheight`, `nbin`, `ticks` |
| `guide_coloursteps()` | `guide_coloursteps(...)` / `guide_colorsteps(...)` | default for `_fermenter`; `even_steps`, `show_limits` |

Not yet: `guide_axis`, `guide_bins`.

<!-- section: facets -->
### Facets

| ggplot2 | ggmanim | notes |
|---|---|---|

Not yet: `facet_wrap`, `facet_grid`, free scales, strips.

<!-- section: coords -->
### Coords

| ggplot2 | ggmanim | notes |
|---|---|---|
| `coord_cartesian(xlim =, ylim =)` | same | zoom without dropping rows; `clip="on"` |
| `coord_flip()`, `coord_fixed(ratio)` | same | |

Not yet: `coord_trans`, `coord_polar`.

<!-- section: themes -->
### Themes

| ggplot2 | ggmanim | notes |
|---|---|---|
| `theme(legend.position = c(1, 0))` | `theme(legend_position=(1, 0))` | dots become underscores |
| `theme_grey()`, `theme_bw()`, `theme_linedraw()`, `theme_light()`, `theme_dark()`, `theme_minimal()`, `theme_classic()`, `theme_void()` | same (+ `theme_gray`) | |
| `theme(axis.text.x = element_text(angle = 45))` | `theme(axis_text_x_angle=45)` | flat fields instead of `element_*` |
| `theme(panel.background = element_rect(fill = ...))` | `theme(panel_background=...)` | |

Not yet: `element_*` objects.

<!-- section: animation -->
### Animation (manim-only)

| ggplot2 / gganimate | ggmanim | notes |
|---|---|---|
| `aes(group = id)` in gganimate | `aes(key="id")` | each layer mobject has `.marks` and matching `.keys` |

Not yet: `GGPlot.create()`, transitions between plots, `ValueTracker`-driven rebuilds.

## Development

```bash
pip install -e '.[dev]'
ruff check . && ruff format --check .
mypy
pytest
manim -ql -s examples/basic.py Basic
```
