# ggplot2 → ggmanim cheat sheet

Everything below is supported; names and arguments follow ggplot2 unless noted.
A section's last line lists what is not there yet, if anything; see [PLAN.md](https://github.com/atzheng/ggmanim/blob/main/PLAN.md) for the
roadmap.

| ggplot2 | ggmanim | notes |
|---|---|---|
| `ggplot(df, aes(x, y, colour = g)) + ...` | `ggplot(df, aes("x", "y", color="g")).add(...)` | strings for columns; `colour` accepted |
| `labs()`, `xlab()`, `ylab()`, `ggtitle()` | `labs()`, `xlab()`, `ylab()`, `labs(title=...)` | |
| `p1 + p2` (patchwork) | build both, arrange the mobjects | see `examples/gallery.py` |

<!-- section: geoms -->
## Geoms

| ggplot2 | ggmanim | notes |
|---|---|---|
| `geom_point()`, `geom_line()`, `geom_path()` | same | `size` = radius in scene units |
| `geom_col()`, `geom_bar(stat = "identity")` | `geom_col()`, `geom_bar(stat="identity")` | stacked by default; plain `geom_bar()` counts (see Stats) |
| `geom_smooth(method = "lm")` | `geom_smooth(method="lm", degree=1)` | polynomial fits, SE ribbon |
| `geom_function(fun = f)` | `geom_function(f)` | |
| `geom_hline()`, `geom_vline()` | same | take part in axis limits |
| `geom_text(aes(label = n))`, `geom_label()` | `geom_text(aes(label="n"))`, `geom_label()` | `size=3.88` = `theme.axis_text_size`; `hjust`/`vjust` incl. `"inward"`; `angle` in degrees; `nudge_x`/`nudge_y`; `check_overlap`; `parse=True` → `MathTex` (tex engine) |
| `annotate("text", x, y, label = ...)` | `annotate("text", x=..., y=..., label=...)` | any geom (`"point"`, `"vline"` ...); never inherits aesthetics |
| `directlabels` package | `geom_line(label=[...])` | direct labels on line/path/function/smooth/vline/hline, drawn like `geom_text` |
| `geom_ribbon(aes(ymin, ymax))`, `geom_area()` | same | `outline_type`; area stacks by default (`position="stack"`), groups aligned on a shared x grid by `stat_align` |
| `geom_rect()`, `geom_tile()` | same | tile `width`/`height` default to the data resolution |
| `geom_raster()` | same | one mobject per fill colour on a regular grid (fast heatmaps) |
| `geom_polygon(aes(group, subgroup))` | same | `subgroup` rings are holes (even-odd fill) |
| `geom_step(direction = "hv")` | same | `"hv"`, `"vh"`, `"mid"` |
| `geom_segment(arrow = arrow())`, `geom_curve(curvature, angle, ncp)` | same | `arrow(length=)` in scene units; curves follow grid's `curveGrob`, shaped in scene space |
| `geom_abline(slope, intercept)` | same | trains no scale; slope/intercept in transformed units (as ggplot2) |
| `geom_errorbar()`, `geom_errorbarh()`, `geom_linerange()`, `geom_pointrange()`, `geom_crossbar()` | same | `orientation="y"` (or `xmin`/`xmax`) for horizontal |
| `geom_boxplot(outlier.colour, notch, varwidth, staplewidth)` | `geom_boxplot(outlier_color=..., notch=True, varwidth=True, staplewidth=0.5)` | dodged by `position_dodge2`; `outliers=False` hides them; horizontal when `y` is discrete |
| `geom_violin(scale, trim, draw_quantiles)` | same | `scale="area"`/`"count"`/`"width"`; dodged by `position_dodge` |
| `geom_density(adjust, bw, kernel, bounds)` | same | outline only unless `fill` is set; `alpha` is the fill's |
| `geom_rug(sides = "bl", length = unit(0.03, "npc"))` | `geom_rug(sides="bl", length=None)` | `length` in scene units (default 3% of the panel); `outside=True` |

Large layers: above `theme(fast_threshold=2000)` rows per panel, `geom_point`,
`geom_rect`/`geom_tile`, `geom_col`/`geom_bar` and `geom_segment` draw *batched*
marks (one `VMobject` per style, every mark a subpath; a 10k-point scatter builds
in well under a second). `fast=True`/`fast=False` forces it either way. Batched
layers look the same, but `layer.keys` is `None`, so per-mark animation and
`plot.mark()` are not available. Mapping `key` turns the automatic switch off.
See `examples/large_scatter.py`.

Not yet: `geom_dotplot`, `geom_spoke`, `geom_quantile`, `geom_blank`, `geom_sf`/`geom_map`.

<!-- section: stats -->
## Stats

| ggplot2 | ggmanim | notes |
|---|---|---|
| `stat_identity()`, `stat_smooth()`, `stat_function()` | same | `after_stat("var")` maps computed variables |
| `geom_smooth(method = "loess", span = .75)` | `geom_smooth(method="loess", span=0.75)` | matches R's `loess` (numpy only); `method="auto"`: loess below 1000 rows, else lm |
| `geom_smooth(method = lm, method.args = ...)`, `fullrange = TRUE` | `geom_smooth(method=f, method_args={...})`, `fullrange=True` | `f(x, y, **args)` returns `predict(xgrid) -> fit` or `(fit, se)` |
| `stat_ecdf(pad = TRUE)` | `stat_ecdf(pad=True)` | step line from the panel's left edge (0) to its right edge (1) |
| `geom_qq()`, `geom_qq_line()`, `stat_qq(distribution = qt, dparams = list(df = 5))` | same; `distribution=scipy.stats.t.ppf, dparams={"df": 5}` | normal by default; the line passes through the quartiles |
| `geom_count()`, `stat_sum()`, `stat_unique()` | same | `size=after_stat("n")` by default; `after_stat("prop")` too |
| `geom_bar()`, `stat_count()` | same | counts rows per x (strings per level); `y=after_stat("count")`/`"prop"`; `weight` aesthetic; stacked by default |
| `geom_histogram(binwidth, bins, center, boundary, breaks, closed, pad)`, `stat_bin()` | same | bin edges as ggplot2's; `after_stat("density")`, `"ncount"`, `"ndensity"`; bins in transformed space on log x; warns when `bins = 30` is defaulted |
| `geom_freqpoly(binwidth = 1)` | same | `stat_bin(pad=True)` drawn as a path, one per group |
| `geom_bin_2d(bins = 30)`, `stat_bin_2d()` | same (`geom_bin2d` too) | tiles, `fill=after_stat("count")`; breaks as ggplot2 4.0 (`binwidth`, `breaks`, `boundary`, `center`, pairs per axis) |
| `geom_hex(bins = 30)`, `stat_bin_hex()` | same | hexbin lattice; hexagons tessellate exactly in data units |
| `geom_density_2d()`, `geom_density_2d_filled()` | same (`geom_density2d` too) | `MASS::kde2d` (`n=100`, `h`, `adjust`); `contour_var`; `contour=False` gives the grid |
| `geom_contour(aes(z = z))`, `geom_contour_filled()` | `geom_contour(aes(z="z"))`, `geom_contour_filled()` | numpy marching squares; `bins`/`binwidth`/`breaks`; filled `level` is an ordered category `"(lo, hi]"` |
| `stat_summary(fun.data = mean_se)` | `stat_summary(fun_data="mean_se")` | default geom `pointrange`; also `mean_cl_normal`, `mean_sdl`, `median_hilow`, `fun`/`fun_min`/`fun_max`; `orientation="y"`; `stat_summary_bin()` |
| `stat_align()` | same | `geom_area`'s default stat: interpolates every group onto the union of x so stacks have no gaps |
| `stat_boxplot(coef = 1.5)` | same | quantile type 7, whiskers to the last point within `coef * IQR`; `weight` gives weighted quantiles |
| `stat_density(bw = "nrd0", adjust, kernel, n = 512, bounds)` | same; `bounds=(0, np.inf)` | matches R's `density()` (numpy FFT); `bw` `"nrd0"`, `"nrd"`/`"scott"`, a number or a callable; `after_stat("count")`, `"scaled"` |
| `stat_ydensity(scale, trim, quantiles)` | same | the violins' densities, plus `violinwidth` |

Not yet: `orientation="y"` for `stat_count`/`stat_bin`; `bw = "ucv"`/`"bcv"`/`"SJ"`.

<!-- section: positions -->
## Positions

| ggplot2 | ggmanim | notes |
|---|---|---|
| `position = "identity"`, `position_identity()` | same | the default for most geoms |
| `position_stack(vjust, reverse)`, `position_fill()` | same | default for `geom_col`/`geom_bar`; first group on top, negatives stacked below 0 |
| `position_dodge(width, preserve)` | same | `preserve="single"` keeps equal bar widths |
| `position_dodge2(width, preserve, padding)` | same | dodges by interval overlap (variable widths) |
| `position_jitter(width, height, seed)`, `geom_jitter()` | same | numpy `default_rng(seed)`; `seed=None` is derived from the data, so builds are reproducible |
| `position_jitterdodge(jitter.width, jitter.height, dodge.width)` | `position_jitterdodge(jitter_width, jitter_height, dodge_width)` | needs a mapped fill/colour/... |
| `position_nudge(x, y)` | same | offsets in transformed units |

Positions accept a name (`position="dodge"`), a name with arguments
(`position="dodge(width=0.8)"`), a class or an instance
(`position=position_dodge(width=0.8)`). See `examples/positions.py`.

<!-- section: scales -->
## Scales

| ggplot2 | ggmanim | notes |
|---|---|---|
| `scale_x_continuous()`, `scale_x_log10()`, `scale_x_sqrt()`, `scale_x_reverse()` | same (`x` and `y`) | `position="top"`/`"right"` |
| `sec_axis(~ . * 1.8 + 32, name)`, `dup_axis()` | `scale_y_continuous(sec_axis=sec_axis(lambda c: c * 1.8 + 32, name="°F"))`, `dup_axis()` | opposite side of the panel; monotone `transform`; breaks use the primary transform, labels its formatter; `derive()`; outer facet panels only; works with `coord_flip` |
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

Not yet: binned position scales, secondary axes on date / discrete scales.

<!-- section: guides -->
## Guides

| ggplot2 | ggmanim | notes |
|---|---|---|
| `guides(colour = "none")` | `guides(color="none")` | |
| `guide_legend()` | `guide_legend(title, nrow, ncol, byrow, reverse, override_aes, order, direction)` | |
| `guide_colourbar()` | `guide_colorbar(...)` / `guide_colourbar(...)` | `barwidth`, `barheight`, `nbin`, `ticks` |
| `guide_coloursteps()` | `guide_coloursteps(...)` / `guide_colorsteps(...)` | default for `_fermenter`; `even_steps`, `show_limits` |

Not yet: `guide_axis`, `guide_bins`.

<!-- section: facets -->
## Facets

| ggplot2 | ggmanim | notes |
|---|---|---|
| `facet_null()` | same | the default: one panel |
| `facet_grid(rows ~ cols, scales, space, margins, switch)` | `facet_grid(rows="drv", cols="cyl")` / `facet_grid("drv ~ cyl")` | rows/cols take a name, list, `"a + b"` or `"."`; free scales are per column (x) / row (y); `space="free_x"` sizes columns by their range; `margins=True` adds `"(all)"` panels; `switch="x"/"y"/"both"` |
| `facet_wrap(~class, nrow, ncol, scales, labeller, dir, as.table, drop, strip.position)` | `facet_wrap("class", nrow=, ncol=, scales=, labeller=, dir=, as_table=, drop=, strip_position=)` | also `["a", "b"]`, `"~ a + b"`, computed `{"big": lambda d: d.x > 4}`; fixed scales put tick labels on the bottom panel of each column / left of each row |
| `label_value`, `label_both`, `labeller(var = ...)` | same | strip labellers; `labeller(am={0: "auto"}, cyl=label_both)` |
| `theme(panel.spacing, strip.background, strip.text, strip.placement)` | `theme(panel_spacing=, panel_spacing_x/_y=, strip_background=, strip_text_size=, strip_text_color=, strip_placement=)` | `panel_width`/`panel_height` size the whole grid |
| faceted plot object | `plot.panels[(row, col)]`, `plot.strips`, `plot.layers[name].panel(row, col)` | `plot.panel`/`plot.axes` are the first panel; custom facets subclass `ggmanim.facets.Facet` |
| layer without the facet variable | same | repeated in every panel (reference / background layers) |
| reveal panels one by one (manim-only) | `for rc, panel in plot.panels.items(): self.play(FadeIn(panel), FadeIn(plot.panel_grid.panel_strips[rc]))` | see `examples/gallery_facets.py` `PanelReveal` |

<!-- section: coords -->
## Coords

| ggplot2 | ggmanim | notes |
|---|---|---|
| `coord_cartesian(xlim =, ylim =)` | same | zoom without dropping rows; `clip="on"` |
| `coord_flip()`, `coord_fixed(ratio)` | same | |
| `coord_trans(x = "log10", y = "sqrt", xlim, ylim)` | `coord_trans(x="log10", y="sqrt", xlim=, ylim=)` | transforms at draw time: stats fit untransformed data, breaks are the scale's placed at transformed positions, lines/polygons/rects are munched into curves |
| `coord_munch(coord, data, range)` | `coord_munch(panel, x, y, closed=False)` | interpolate a path for a non-linear coord (`panel.is_linear` false); geoms use `geoms.base.munch_path` |
| `coord_polar(theta = "x", start = 0, direction = 1)` | `coord_polar(theta="x", start=0, direction=1, inner_radius=0)` | bars are munched into wedges; `geom_bar(width=1)` + `coord_polar(theta="y")` is a pie, `geom_col(width=1)` + `coord_polar()` a coxcomb/rose; theta labels around the circle, radius axis on the left; never clips |
| donut: `xlim(c(0.5, 2.5))` with bars at `x = 2` | same (`xlim(0.5, 2.5)`), or `coord_polar(theta="y", inner_radius=0.5)` | `inner_radius` is a fraction of the outer radius (like `coord_radial(inner.radius=)`) |
| radar chart (`geom_polygon(fill = NA)` + a straight-edged polar coord) | `geom_polygon(fill=None)` + `coord_radar()` | `coord_polar` mapping, but edges stay straight (not munched); add `ylim(0, max)` so the centre is 0 |

<!-- section: themes -->
## Themes

| ggplot2 | ggmanim | notes |
|---|---|---|
| `theme(legend.position = c(1, 0))` | `theme(legend_position=(1, 0))` | dots become underscores |
| `theme_grey()`, `theme_bw()`, `theme_linedraw()`, `theme_light()`, `theme_dark()`, `theme_minimal()`, `theme_classic()`, `theme_void()` | same (+ `theme_gray`) | |
| `theme(axis.text.x = element_text(angle = 45))` | `theme(axis_text_x_angle=45)` | flat fields instead of `element_*` |
| `theme(panel.background = element_rect(fill = ...))` | `theme(panel_background=...)` | |

Not yet: `element_*` objects.

<!-- section: animation -->
## Animation (manim-only)

| ggplot2 / gganimate | ggmanim | notes |
|---|---|---|
| `aes(group = id)` in gganimate | `aes(key="id")` | each layer mobject has `.marks` and matching `.keys` |
| `enter_grow()` / `enter_fade()` | `self.play(plot.create())` | panel, axes, each layer, legend, titles; `lag_ratio`, `run_time`, `order=`, `panel_lag` |
| — | `plot.reveal_layer("fit")`, `plot.hide_layer("fit")` | the geom's own animation; `animation="fade_in"` overrides |
| — | `plot.mark("points", "Ohio")` | the mobject drawn for the row whose `key` is `"Ohio"` |

Per-geom animations live in a registry (`ggmanim.animate.register_animation`): points
fade in, lines and axes are drawn (`Create`), bars grow out of the zero line
(sideways under `coord_flip`), areas/tiles/polygons fade in and text is written.
The built plot records which geom drew each layer (`plot.layer_geoms`); a layer
whose geom is unknown gets an animation inferred from its marks.

Transitions between plots live in `ggmanim.transition`:

| gganimate | ggmanim | notes |
|---|---|---|
| tweening two plots | `self.play(transition(p1, p2, key="country"))` | marks matched by `key`: matched ones `Transform`, the rest fade out / in; `t.plot` is the built second plot |
| `view_static()` | `transition(..., limits="union")` | shared position limits (`"a"`, `"b"`, `{"x": (lo, hi)}`); default: the axes rescale |
| `transition_states(year)` | `animate_states(spec, "year", key="country", title="Year {state}")` | one plot per value; `transition_length`, `state_length`, `wrap` |
| `transition_reveal(x)` | `animate_reveal(spec, along="x", steps=10)` | lines grow, points appear, along a column |

```python
import pandas as pd
from ggmanim import aes, geom_point, ggplot
from ggmanim.transition import transition

df52 = pd.DataFrame({"country": ["A", "B", "C"], "gdp": [1, 2, 3], "life": [40, 50, 45]})
df07 = pd.DataFrame({"country": ["B", "C", "D"], "gdp": [4, 6, 5], "life": [60, 70, 65]})
p52 = ggplot(df52, aes("gdp", "life")).add(geom_point(), name="points")
p07 = ggplot(df07, aes("gdp", "life")).add(geom_point(), name="points")
t = transition(p52, p07, key="country")  # B and C move, A fades out, D fades in
assert t.layer_stats["points"] == {"matched": 2, "exited": 1, "entered": 1}
```

The second plot is built with the first one's panel size and aligned onto it.
Axis lines morph; tick labels, ticks and grid lines are matched by label text, so
shared ticks glide to their new place and the others fade while sliding with the
axis. Legends and titles transform when compatible, else cross-fade. When the
animation ends the scene holds `t.plot`, so transitions chain. See
[`examples/transitions.py`](https://github.com/atzheng/ggmanim/blob/main/examples/transitions.py) (scatter between years,
re-sorting bars, a looping 4-state GIF, a line reveal).

Not yet: `ValueTracker`-driven rebuilds.
