# Scales & guides

## Default scales

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

## Limits vs zoom

As in ggplot2, scale limits (`xlim(0, 7)`,
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

## Secondary axes

`sec_axis()` adds a second axis on the opposite side of the panel, a monotone
transform of the primary one; `dup_axis()` copies it. `examples/sec_axis.py`
has a scene with each, also under `coord_flip` and in facets:

```python
from ggmanim import scale_y_continuous, sec_axis

scale_y_continuous(sec_axis=sec_axis(lambda c: c * 1.8 + 32, name="°F"))
```
