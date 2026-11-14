# Animation

A built plot is an ordinary manim `VGroup`, so any manim animation works on it
or on its parts. On top of that, ggmanim has two modules:

- [`ggmanim.animate`](../api/animate.md): reveal a plot stage by stage with an
  animation chosen per geom, and address single marks by key;
- [`ggmanim.transition`](../api/transition.md): morph one plot into another,
  mark by mark, after gganimate's `transition_*`;
- [`ggmanim.live`](../api/live.md): update a built plot in place every frame
  (updaters, `ValueTracker`s).

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

## Live plots

A transition morphs between two finished builds. Live plots
([`ggmanim.live`](../api/live.md)) instead update one built plot every frame,
from an updater or a `ValueTracker`, without rebuilding its axes, tick labels or
legend. The panel is frozen at the first build, so the position limits must be
fixed (`xlim`/`ylim`, scale `limits=`, or `coord_cartesian(xlim=, ylim=)`).

| ggmanim | notes |
|---|---|
| `p.with_data(df)` | immutable copy of the spec with new plot data |
| `plot = build_live(p)`; `plot.update_data(df)` | marks move in place (same objects, keyed by `key`); identity stats only; use inside `add_updater` |
| `always_rebuild(p)` | `always_redraw` with a cached panel: layers recomputed every frame, e.g. `geom_function(fun=lambda x, a=tracker: ...)` |
| `LivePlot(lambda: spec)` | holds the spec (or a spec factory); `.refresh()`, `.update_data(df)`, `.updater()` |

```python
import numpy as np
import pandas as pd
from manim import ValueTracker
from ggmanim import aes, geom_point, ggplot
from ggmanim.live import always_rebuild, build_live

df52 = pd.DataFrame({"country": ["A", "B", "C"], "gdp": [1, 2, 3], "life": [40, 50, 45]})
pts = ggplot(df52, aes("gdp", "life")).add(geom_point(), name="points").xlim(0, 8).ylim(30, 80)
live = build_live(pts)
dot = live.mark("points", 0)
live.update_data(df52.assign(life=[45, 55, 50]))  # moves the dots
assert live.mark("points", 0) is dot

a = ValueTracker(1.0)
wave = ggplot().geom_function(fun=lambda x, a=a: a.get_value() * np.sin(x)).xlim(0, 6.3).ylim(-2, 2)
curve = always_rebuild(wave)  # self.add(curve); self.play(a.animate.set_value(2))
```

`update_data` keeps the mark objects (points and text are moved, lines and bars
take the new outline), so `plot.mark(...)` references and updaters stay valid.
When rows are added or removed, that layer's mobject is replaced instead, still
without touching any text. `examples/live_plot.py`:

```python
--8<-- "examples/live_plot.py"
```
