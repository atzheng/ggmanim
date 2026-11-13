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

## Legends

Every mapped non-position aesthetic gets a guide: a legend of keys for discrete
scales (and size/alpha/linewidth), a colourbar for continuous colour. As in
ggplot2, legends with the same title and the same levels merge into one with
composite keys, so `color="g", shape="g"` gives a single legend. Configure a
guide with `guide_legend(...)` / `guide_colorbar(...)`, either in `guides()` or
as a scale's `guide=`; place and style them with `theme()`:

```python
import numpy as np
import pandas as pd
from ggmanim import aes, ggplot, guide_colorbar, guide_legend, guides, theme

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
```

- `guide_legend(title, nrow, ncol, byrow, reverse, override_aes, order, direction)`;
  `guide_colorbar(title, barwidth, barheight, nbin, reverse, ticks, order, direction)`.
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

## What is supported

Phase 1 (this release): `geom_point`, `geom_line`, `geom_smooth` (`method="lm"`,
polynomial `degree=`, SE ribbon), `geom_col`/`geom_bar` (identity heights),
`geom_function`, `geom_vline`, `geom_hline`; continuous x/y scales with pretty
breaks, 5% expansion, `xlim`/`ylim`, `scale_*_log10`/`sqrt`/`reverse`; discrete
colour/fill/shape/linetype scales (`scale_color_manual`, `scale_fill_identity`,
...) with legends; `coord_cartesian`; `theme_default`/`theme_minimal`/`theme_classic`
and `theme(...)`; `labs`, `xlab`, `ylab`, `guides`.

See [PLAN.md](PLAN.md) for the roadmap (facets, stats, positions, colourbars,
transitions between plots).

## Development

```bash
pip install -e '.[dev]'
ruff check . && ruff format --check .
mypy
pytest
manim -ql -s examples/basic.py Basic
```
