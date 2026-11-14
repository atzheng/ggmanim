# Getting started

## Install

```bash
pip install git+https://github.com/atzheng/ggmanim                    # manim, pandas, numpy
pip install "ggmanim[stats] @ git+https://github.com/atzheng/ggmanim"  # + scipy
```

ggmanim needs Python 3.10+ and a working manim installation (cairo and pango;
see manim's [installation guide](https://docs.manim.community/en/stable/installation.html)).
LaTeX is *not* needed: text is drawn with Pango by default. The `stats` extra
(`scipy`) is optional; it gives exact t-quantiles for `geom_smooth` confidence
bands.

## A first plot

A plot is a spec: data, a default aesthetic mapping, and components added with
`.add(...)`. `.build()` turns the spec into a manim mobject.

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

Every registered component also has a chaining method, so
`p.geom_point(alpha=0.8)` is the same as `p.add(geom_point(alpha=0.8))`.

## A first scene

The built plot is a `VGroup` with named parts (`plot.panel`, `plot.layers[name]`,
`plot.legend`, `plot.title`, ...), so a scene can animate them in any order.
This is `examples/basic.py`:

```python
--8<-- "examples/basic.py"
```

Render a still of the last frame, or the animation:

```bash
manim -ql -s examples/basic.py Basic
manim -ql examples/basic.py Basic
```

Instead of animating the parts by hand, `self.play(plot.create())` reveals the
whole plot in a sensible order with an animation chosen per geom; see the
[animation guide](concepts/animation.md).

## Next steps

- [Concepts](concepts/index.md) explains the handful of ways ggmanim differs from ggplot2.
- [The grammar](concepts/grammar.md) walks through each component.
- Every file in [`examples/`](https://github.com/atzheng/ggmanim/tree/main/examples)
  is a runnable scene; `manim -ql -s -a examples/<file>.py` renders all of its scenes.
