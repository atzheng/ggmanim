# Geoms, stats & positions

A layer combines three of the grammar's components: the **stat** computes what
to draw from the data, the **position** resolves overlaps, and the **geom**
draws the result through the coord. Every `geom_*` has a default stat and
position, and every `stat_*` a default geom; both can be overridden:

```python
geom_bar()                                  # stat="count", position="stack"
geom_bar(stat="identity")                   # = geom_col(): heights from y
geom_col(position="dodge")                  # side by side
stat_summary(fun_data="mean_se")            # geom="pointrange"
geom_text(aes(label=after_stat("count")), stat="count")
```

The [cheat sheet](../cheatsheet.md) lists every geom, stat and position with
its ggplot2 counterpart; the [geom](../api/geoms.md), [stat](../api/stats.md)
and [position](../api/positions.md) references document their arguments.

## Geoms

Geoms come in families:

| family | geoms |
|---|---|
| points | `geom_point`, `geom_jitter`, `geom_count` |
| lines | `geom_line`, `geom_path`, `geom_step`, `geom_function`, `geom_smooth`, `geom_freqpoly`, `geom_qq_line` |
| reference lines | `geom_hline`, `geom_vline`, `geom_abline` |
| bars and areas | `geom_col`, `geom_bar`, `geom_histogram`, `geom_area`, `geom_ribbon`, `geom_density` |
| rectangles and polygons | `geom_rect`, `geom_tile`, `geom_raster`, `geom_polygon`, `geom_bin_2d`, `geom_hex`, `geom_contour_filled`, `geom_density_2d_filled` |
| intervals | `geom_errorbar`, `geom_errorbarh`, `geom_linerange`, `geom_pointrange`, `geom_crossbar` |
| distributions | `geom_boxplot`, `geom_violin`, `geom_rug`, `geom_qq` |
| text and annotation | `geom_text`, `geom_label`, `annotate`, `geom_segment`, `geom_curve` |

Things to know:

- `size` is a radius and `linewidth` a manim stroke width (see
  [scene units](index.md#sizes-are-scene-units)).
- `geom_hline`/`geom_vline`/`geom_function` take part in the automatic axis
  limits, so reference lines are never off-panel, and are repeated in every facet.
- Lines, paths, functions, smooths and reference lines take `label=` for
  direct labels at their end (`geom_line(label=["cos", "sin"])`).
- Every geom attaches `.marks` and `.keys` to the mobject it draws, so single
  marks can be animated (see [Animation](animation.md)), except for batched
  large layers (see [Performance](performance.md)).

`examples/annotations.py` puts text on bars, labels on points, and uses
`annotate()`:

```python
import pandas as pd
from ggmanim import aes, geom_col, geom_text, ggplot, labs

levels = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
counts = pd.DataFrame({"cut": pd.Categorical(levels, categories=levels)})
counts["n"] = [16, 49, 121, 138, 215]

plot = (ggplot(counts, aes("cut", "n"))
        .add(geom_col(fill="#4C72B0"), name="bars")
        .add(geom_text(aes(label="n"), vjust=0, nudge_y=4), name="counts")
        .add(labs(title="Counts on bars", x="Cut", y="Diamonds"))
        .build())
```

## Stats

Stats are pure pandas/numpy (no manim) and run per group, per panel, in the
transformed space of the position scales (so `geom_histogram` on a log x axis
bins in log space). Variables a stat computes are mapped with `after_stat`.
From `examples/histograms.py`:

```python
import numpy as np
import pandas as pd
from ggmanim import aes, after_stat, geom_freqpoly, geom_histogram, ggplot, labs

rng = np.random.default_rng(7)
df = pd.DataFrame({
    "height": np.concatenate([rng.normal(164, 7, 300), rng.normal(177, 8, 300)]),
    "sex": np.repeat(["F", "M"], 300),
})

stacked = ggplot(df, aes("height", fill="sex")).add(
    geom_histogram(binwidth=2.5, color="#222222"), labs(fill="Sex")).build()
polygons = ggplot(df, aes("height", color="sex")).add(geom_freqpoly(binwidth=2.5)).build()
density = ggplot(df, aes("height")).add(
    geom_histogram(aes(y=after_stat("density")), bins=25, fill="#4C78A8")).build()
```

The smoothers (`method="lm"`, `"loess"`, a custom callable), densities and
2-D densities follow R's numerics (`loess`, `density()`, `MASS::kde2d`)
without needing R or scipy. See `examples/smoothers.py`,
`examples/distributions.py` and `examples/two_d.py`.

## Positions

Positions adjust the drawn data after the stat and the geom's `setup_data`, so
they see bar extents:

- `position_stack(vjust, reverse)` (default for bars and areas; negatives stack
  below zero), `position_fill()` (stack to 1);
- `position_dodge(width, preserve)` and `position_dodge2(padding)` (boxplots);
- `position_jitter(width, height, seed)` and `position_jitterdodge()`; jitter
  is reproducible (`seed=None` derives a seed from the data);
- `position_nudge(x, y)`.

`examples/positions.py`:

```python
--8<-- "examples/positions.py"
```
