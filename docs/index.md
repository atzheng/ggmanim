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

```python
from manim import Scene
from ggmanim import aes, geom_point, geom_smooth, ggplot, labs, scale_x_log10

class DoseResponse(Scene):
    def construct(self):
        plot = (ggplot(df, aes("dose", "response", color="compound"))
                .add(geom_point(alpha=0.8), name="points")
                .add(geom_smooth(method="lm"), name="fit")
                .add(scale_x_log10(), labs(x="Dose (mg)", y="Response"))
                .build())
        self.play(plot.create())              # panel, axes, layers, legend, titles
```

## Where to go next

- **[Getting started](getting-started.md)**: install, the first plot, the first scene.
- **[Concepts](concepts/index.md)**: what is different from ggplot2 (immutable
  specs, the build pipeline, scene units, text engines, foreground colours).
- **[The grammar](concepts/grammar.md)**: aesthetics, layers, geoms, stats,
  positions, scales, coords, facets and themes, one by one.
- **[Animation](concepts/animation.md)**: ordered reveals, per-geom animations,
  data-keyed transitions between plots.
- **[Performance](concepts/performance.md)**: the batched fast path for large layers.
- **[Cheat sheet](cheatsheet.md)**: every ggplot2 function and its ggmanim spelling.
- **[API reference](api/index.md)**: generated from the docstrings of the public modules.
