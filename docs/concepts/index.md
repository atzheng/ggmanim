# Concepts

ggmanim follows ggplot2's names and argument semantics. This page covers the
few places where it differs, because it is Python and because the result is a
manim mobject rather than a static image.

## The spec is immutable

A `ggplot` is a description of a plot: data, default mapping, layers, scales,
coord, facet, theme and labels. There is no `+`: components are combined with
`.add(...)`, which returns a **new** plot. The DataFrame is shared by
reference, not copied.

```python
base = ggplot(df, aes("x", "y"))
points = base.add(geom_point())             # base is unchanged
both = points.add(geom_smooth(method="lm"))
```

Because specs never change, you can branch from a common base, keep several
variants around, and build the same spec as often as you like.

Every registered component also has a chaining method generated from the
registry, so these two lines are equivalent:

```python
p.add(geom_point(alpha=0.7))
p.geom_point(alpha=0.7)
```

`.add(..., name="points")` names the single layer being added, so you can look
it up after the build as `plot.layers["points"]`.

Column names are strings: `aes("x", "y", color="group")`. A callable
(`aes(y=lambda d: d.a / d.b)`) computes a column, and `after_stat("count")`
refers to a variable computed by the stat. `colour` and `color` are both
accepted everywhere.

## The build pipeline

`p.build()` runs a pure pipeline, spec → [`GGPlot`](../api/mobjects.md). It
mirrors ggplot2's `ggplot_build` + `ggplot_gtable`, one function per step in
[`ggmanim.build`](../api/extending.md):

1. resolve layers: layer data and mapping (`inherit_aes`), evaluate aesthetics, add groups;
2. assign panels: the facet layout and each row's `PANEL`;
3. set up position scales (trained on the raw data), then compute stats;
4. train and map the non-position aesthetics (colour, fill, size, shape, ...);
5. `geom.setup_data` (for example bar extents);
6. apply positions (stack, dodge, jitter, ...) and retrain the position scales;
7. build the panel(s) through the coord;
8. draw every layer with `geom.draw_panel`;
9. assemble legends, titles and the `GGPlot`.

The pipeline never mutates the spec: scales are cloned per build and layer data
is copied. Stats and scales are pure pandas/numpy and never import manim; only
steps 7–9 make mobjects.

The result is a `VGroup` with named parts:

| attribute | what |
|---|---|
| `plot.panel` | grid, axes, ticks, tick labels, axis titles |
| `plot.axes` | the panel's manim `Axes` |
| `plot.layers[name]` | one mobject per layer; `.marks` and `.keys` address single marks |
| `plot.legend` | the placed legends (empty `VGroup` when there are none) |
| `plot.title`, `plot.subtitle`, `plot.caption` | `Mobject` or `None` |
| `plot.panels[(row, col)]`, `plot.strips` | facet panels and strips |
| `plot.scales`, `plot.layer_data` | the trained scales and the data each geom drew |

## Sizes are scene units

There are no millimetres or points. Sizes are manim scene units (the default
frame is about 14.2 × 8 units): `geom_point(size=0.08)` is a dot radius,
`theme(panel_width=6, panel_height=4)` sets the panel size, and
`arrow(length=0.15)` is an arrowhead length. Line widths are manim stroke
widths (`linewidth=2`). Text sizes in geoms keep ggplot2's scale
(`geom_text(size=3.88)` is the theme's axis text size).

## Text vs Tex

Titles, tick labels and legend text go through `theme.make_text`, which uses
Pango (`manim.Text`) by default, so **no LaTeX installation is needed**. To
typeset them with LaTeX instead (requires LaTeX, as for manim's `Tex`):

```python
from ggmanim import ggplot, theme

tex_plot = ggplot().geom_function(lambda x: x ** 2, xlim=(0, 2)).add(
    theme(text_engine="tex")).labs(x=r"$x$", y=r"$x^2$")
```

Any label may also be a ready-made manim `Mobject`, e.g.
`labs(y=MathTex(r"\hat{y}"))`. `label_math()` writes `10^k` with Unicode
superscripts under the Text engine and as LaTeX under `text_engine="tex"`.
`geom_text(parse=True)` renders labels as `MathTex` (tex engine).

## Foreground and background

The default theme (`theme_default()`, also `theme_manim()`) is drawn straight
on the scene background, and its foreground colour follows manim's
`config.background_color` at build time: white axes and text on the default
black background, black on a white one. The ggplot2 looks (`theme_gray()`,
`theme_bw()`, `theme_linedraw()`, `theme_light()`, `theme_dark()`) bring their
own white plot background; `theme_minimal()`, `theme_classic()` and
`theme_void()` are background-agnostic. See [Themes](themes.md).
