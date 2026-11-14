# Animation

A built plot is an ordinary manim `VGroup`, so any manim animation works on it
or on its parts. On top of that, ggmanim has two modules:

- [`ggmanim.animate`](../api/animate.md): reveal a plot stage by stage with an
  animation chosen per geom, and address single marks by key;
- [`ggmanim.transition`](../api/transition.md): morph one plot into another,
  mark by mark, after gganimate's `transition_*`.

## Animating the parts by hand

`plot.panel`, `plot.layers[name]`, `plot.legend` and `plot.title` are separate
mobjects, so a scene can show them in any order with any animation:

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

## `plot.create()`: an ordered reveal

`plot.create()` returns one animation that reveals the panel, the axes, each
layer (with its geom's own animation), the legend and the titles in turn.
`lag_ratio`, `run_time` and `order=` control the sequence, `panel_lag` staggers
facet panels and `mark_lag` the marks within a layer.

| method | what |
|---|---|
| `plot.create(lag_ratio=1.0, run_time=None, order=(...), panel_lag=, mark_lag=, animations={layer: preset})` | the whole plot, stage by stage |
| `plot.reveal_layer("fit", animation=None)` | one layer, with its geom's animation (or `"fade_in"`, `"create"`, ...) |
| `plot.hide_layer("fit")` | the reverse |
| `plot.mark("points", "Ohio")` | the mobject drawn for the row whose `key` is `"Ohio"` |

How a layer appears depends on its geom: points fade in, lines and axes are
drawn (`Create`), bars grow out of the zero line (sideways under `coord_flip`),
areas, tiles and polygons fade in, and text is written. The built plot records
which geom drew each layer (`plot.layer_geoms`); new geoms register their own
animation with `ggmanim.animate.register_animation`.

`examples/animate_create.py`:

```python
--8<-- "examples/animate_create.py"
```

```bash
manim -ql examples/animate_create.py AnimateCreate
manim -ql examples/animate_create.py AnimateFacets
```

## Keys: addressing single marks

Map the `key` aesthetic to give every mark an identity: `aes(key="region")`.
Each drawn layer then has `.marks` and matching `.keys`, `plot.mark(layer, key)`
finds a mark, and transitions match marks across plots by key. Without `key`,
marks are identified by data row index (or group id for lines).

```python
self.play(Indicate(plot.mark("bars", "East")))
```

## Transitions between plots

`transition(p1, p2, key=...)` morphs plot `p1` into plot `p2`: marks with the
same key `Transform`, marks only in `p1` fade out and marks only in `p2` fade
in. The second plot is built with the first one's panel size and aligned onto
it. Axis lines morph; tick labels, ticks and grid lines are matched by label
text, so shared ticks glide to their new place. Legends and titles transform
when compatible, else cross-fade. When the animation ends the scene holds
`t.plot`, the built second plot, so transitions chain:
`transition(t.plot, p3, ...)`.

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

| gganimate | ggmanim | notes |
|---|---|---|
| tweening two plots | `self.play(transition(p1, p2, key="country"))` | `t.plot` is the built second plot |
| `view_static()` | `transition(..., limits="union")` | shared position limits (`"a"`, `"b"`, `{"x": (lo, hi)}`); default: the axes rescale |
| `transition_states(year)` | `animate_states(spec, "year", key="country", title="Year {state}")` | one plot per value; `transition_length`, `state_length`, `wrap` |
| `transition_reveal(x)` | `animate_reveal(spec, along="x", steps=10)` | lines grow, points appear, along a column |

`examples/transitions.py` moves a scatter between years, re-sorts bars, loops
through four states (a GIF-friendly loop) and reveals lines along x:

```python
--8<-- "examples/transitions.py"
```

```bash
manim -ql examples/transitions.py ScatterYears
manim -ql --format=gif examples/transitions.py StatesLoop
```
