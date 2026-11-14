# ggmanim — implementation plan

A grammar-of-graphics (ggplot2) interface for [manim](https://www.manim.community/).
This plan covers the move from the current single-class prototype to a robust,
performant, idiomatic implementation of as much of ggplot2 as is sensible in manim.

Status of each phase is tracked in beads (`bd list`). Phase 1 is the refactor and
lands first, alone.

**Status (2026-11-13):** Phases 0–2 complete (862 tests; `examples/gallery.py`).
Phases 3–8 are beaded as epics ggm-2m8 (geoms/stats/positions), ggm-913 (facets),
ggm-aji (coords), ggm-bam (performance), ggm-ifn (animation), ggm-xll (docs/release).
Section 1 below describes the pre-Phase-1 prototype and is kept for history.

## 1. Current state (as of 2026-11-13)

`ggmanim/GGManim.py` (504 lines) implements 7 geoms as methods on one mutable
`GGManim` class, a hand-rolled discrete colour scale and a single legend. It works
for the happy path (one DataFrame, numeric x/y, `color` mapped to a categorical
column) but has no stats, positions, scale objects, coords, facets or themes.

### Known defects

| Where | Problem |
|---|---|
| `add_layer` | `data or self.data` raises `ValueError` (DataFrame truthiness) whenever per-layer data is passed. |
| `build` range inference | Reads `layer.aes["x"]`/`["y"]` on every layer; `geom_function`/`vline`/`hline` layers have no x/y aes → `KeyError`/`TypeError` with auto-limits. |
| `auto_breaks` | Step only (`10^round(log10(range/5))`) → 2–16 ticks, no 1-2-5 rounding, no snapping, no expansion. Leftover `print`. |
| `scale_*_log10` + `LogBase` | Manim's `LogBase` expects the range in exponent units; raw data min/max is passed, so log plots are wrong. |
| colour scale | Only `color` is mapped; `fill_color`/`shape`/`linetype`/`size`/`alpha` are passed as raw column values into mobject kwargs → `TypeError`. `CATEGORICAL_AES` unused; `color_mapping` dead. |
| `palettes.HashableManimColor` | Overrides `__len__`/`__iter__`/`__getitem__`/`__str__` to delegate to an `int`; exists only to stop `np.asarray` expanding colours. A list removes the need. |
| `build_geom_line` | Label uses the last group's data only; `group_kwargs["color"]` `KeyError` when grouping by `fill_color`. |
| `build_geom_smooth` | Linear only, no SE ribbon, no `method=`; uses scikit-learn for what `np.polyfit` does. |
| `build_geom_bar` | Identity-only (`geom_col` semantics), no stack/dodge, bars anchored at y=0 even if out of range, manual width math wrong on log x, no categorical x. |
| `build` | `self.y_label.rotate(PI/2)` mutates stored state → second `build()` rotates it 180°. Build is not idempotent. |
| `Axes(...).set_color(BLACK)` | Hard-coded for a white background; manim defaults to black. |
| `__init__.py` | `import *` without `__all__` re-exports `f`, `np`, `pd`, … |
| `setup.py` | `distutils` (gone in 3.12), no `install_requires`; `funcy`/`sklearn` undeclared. |
| repo | No tests/README/examples/lint/CI; stray `ggmanim/gurobi.log`. |

### Worth keeping

- Named, addressable parts of the built plot (`plot["axes"]`, `plot["layers"][id]`,
  `plot["legends"]`) — manim users animate pieces individually.
- Vectorised `axes.plot(..., use_vectorized=True)` for curves.
- Tol "bright"/"muted" palettes (colour-blind safe).

## 2. Target architecture

Mirror ggplot2's build pipeline (`ggplot_build`). Each concept is a small, testable
object; the plot is an **immutable spec**; `build()` is a pure function spec → mobject.

```
ggmanim/
  __init__.py        public API with explicit __all__
  aes.py             aes() mapping; column refs, callables, after_stat()
  plot.py            ggplot spec: data, mapping, layers, scales, coord, facet, theme, labels
  layer.py           Layer = geom + stat + position + data + mapping + params + inherit_aes
  geoms/             Geom base (required_aes, default_aes, draw_group/draw_panel, legend glyph)
  stats/             Stat base (compute_group/compute_panel) — pure numpy/pandas, no manim
  positions.py       identity, stack, fill, dodge, jitter, nudge
  scales/
    base.py          Scale: train / map / breaks / labels / limits / expand / oob
    continuous.py    position, size/alpha, colour gradients; transforms (log10, sqrt, reverse)
    discrete.py      position (categorical x/y); colour/fill/shape/linetype palettes
    breaks.py        pretty / extended (Wilkinson) breaks, log breaks, label formatters
  coords.py          CoordCartesian (owns manim Axes creation + vectorised c2p), flip, fixed
  facets.py          facet_null, facet_wrap, facet_grid → panel layout
  guides.py          Legend (merged per scale, geom key glyphs), colourbar, axis guides
  theme.py           Theme dataclass → manim Axes/Text config; theme_gray/bw/minimal/classic
  palettes.py        plain lists / callables; Tol, viridis, hue, brewer subsets
  build.py           the pipeline (below)
  mobjects.py        GGPlot(VGroup) with .axes/.panels/.layers[id]/.legend/.title
  animate.py         create() ordering, layer-keyed transitions between plots
tests/               unit (no manim render), structural (build → inspect mobjects), golden images
examples/            gallery scripts rendered by CI
```

### Build pipeline (`build.py`), one function per step

1. Resolve each layer's data + mapping (layer overrides, `inherit_aes`); evaluate aes expressions.
2. Facet: assign `PANEL` to rows.
3. Compute stats per group/panel (`identity`, `count`, `bin`, `smooth`, …) — pure pandas/numpy.
4. Train non-position scales on stat output; map `color`/`fill`/`size`/`alpha`/`shape`/`linetype` → manim values.
5. Apply positions (stack/dodge/jitter).
6. Train position scales (incl. transforms) → limits, breaks, labels, expansion.
7. Coord computes panel params and builds one manim `Axes` per panel.
8. Geoms draw: data (already in scale space) + coord → VMobject, vectorised via `axes.c2p(xs, ys)`.
9. Assemble `GGPlot`: panels, strips, axis titles, title/subtitle/caption, legends — laid out by theme.

`build()` never mutates the spec, so it is idempotent and plots can be rebuilt for transitions.

### API: component objects + method chaining (seaborn.objects style)

Components (`geom_point(...)`, `scale_x_log10()`, `theme_minimal()`) are plain objects.
`.add(*components)` is the one real composition method and returns a **new** plot
(the spec is tiny; the DataFrame is shared by reference). Per-component chaining
methods are generated from the component registry as sugar. No `__add__`.

```python
p = (ggplot(df, aes("x", "y", color="group"))
     .add(geom_point(alpha=0.7), geom_smooth(method="lm"))
     .add(scale_x_log10(), theme_minimal())
     .labs(x="Dose", y="Response", title="..."))

# equivalent sugar
p = ggplot(df, aes("x", "y", color="group")).geom_point(alpha=0.7).geom_smooth(method="lm").scale_x_log10()

mob = p.build()                       # GGPlot(VGroup)
self.play(mob.create())               # axes → layers → legend
self.play(mob.layers["pts"].animate.set_opacity(0.2))   # .add(geom_point(), name="pts")
```

`GGManim` stays as an alias of `ggplot`; its existing `.geom_*/.xlab/.ylab/.xlim/.ylim/.scale_*_log10`
chaining keeps working, and the built mobject preserves `["axes"]/["layers"]/["legends"]` access.

### Decisions

1. Composition via `.add()` + generated chaining; immutable spec; no `+` operator.
2. Default text engine is `Text` (Pango), not `Tex`: no LaTeX dependency, ~10× faster builds.
   `theme(text_engine="tex")` switches back; any label may also be a ready-made Mobject.
3. Drop scikit-learn and funcy. `scipy` is an optional extra (`[stats]`) for t-quantiles/KDE/loess;
   fall back to normal approximations / numpy implementations when absent.
4. Log scales are a scale *transform* (train/break in log space, draw on a linear manim `Axes`
   with custom tick labels) rather than `LogBase` — fixes the exponent-range bug and keeps
   `c2p` uniform across geoms.
5. Theme reads `config.background_color` to pick a default foreground colour.
6. No R datasets shipped (licensing/size); small synthetic data in `examples/`.

## 3. Phases

Each phase ends with tests green and at least one example rendered.

### Phase 0 — Project hygiene
- `pyproject.toml`; deps `manim>=0.19`, `pandas>=2`, `numpy`; extra `[stats]` → `scipy`.
- `ruff` (lint+format), `mypy`, `pytest`; GitHub Actions running tests headless (no LaTeX).
- README with install + 3 examples; delete `gurobi.log`; commit the pending `.gitignore` change.
- Pin supported manim range (0.19–0.22 at time of writing).

### Phase 1 — Core refactor to the grammar (functional parity, bugs fixed)
- `aes`, `ggplot`/`.add()`, `Layer`, `Geom`/`Stat`/`Position`/`Scale`/`Coord`/`Theme` bases, the 9-step pipeline.
- Port existing geoms: `point`, `line`, `function`, `vline`, `hline`, `smooth` (lm via `np.polyfit`), `bar`→`col`.
- Continuous x/y scales with pretty breaks, 5% expansion, user limits/breaks; `log10` as a transform.
- Discrete colour/fill with palettes; legend per scale with geom-appropriate key glyphs.
- `GGPlot` mobject with key access preserved; `GGManim` alias with chaining.
- Tests: breaks, scale train/map, each stat, structural tests (point centre == `c2p`), build idempotence,
  per-layer data, mixed `geom_function` + data layers.

### Phase 2 — Scales & guides
- Discrete x/y; `reverse`, `sqrt`; `coord_cartesian(xlim=)` zoom vs `xlim()` censor; `oob`; `expand`; dates later.
- `scale_color_manual/identity/brewer/viridis_d`; continuous `gradient/gradient2/gradientn/viridis_c` + colourbar.
- `fill` separate from `color`; `size`/`alpha` continuous; `shape` markers; `linetype` via `DashedVMobject`.
- Legend merging, `guides()`, `theme(legend_position=...)`, titles; tick-label formatters; `labs()` title/subtitle/caption.

### Phase 3 — Geom/stat breadth (by payoff)
1. `geom_col`, `geom_bar` (stat_count), `geom_histogram` (stat_bin), `position_stack/fill/dodge`.
2. `geom_path/step/segment/abline`, `geom_area/ribbon`, `geom_errorbar/linerange/pointrange/crossbar`, `stat_summary`.
3. `geom_text/label` (Text/Tex/MathTex), `annotate()`, `geom_rect/tile`, `geom_polygon`.
4. `geom_boxplot`, `geom_violin`, `geom_density`, `geom_freqpoly`, `geom_jitter`, `geom_rug`.
5. `geom_smooth`: polynomial `lm` + SE ribbon, `loess`.
6. `geom_density_2d`, `geom_bin2d`, `geom_hex` (low priority).

### Phase 4 — Facets
`facet_wrap`, `facet_grid`; shared/free scales; strip labels; shared legend; `GGPlot.panels[(i, j)]`.

### Phase 5 — Coords
`coord_cartesian`, `coord_flip`, `coord_fixed`, `coord_trans`; `coord_polar` last.

### Phase 6 — Performance
- Fully vectorised stats/scales/positions; one `c2p` call per geom group.
- Fast path for large n: one `VMobject` with many subpaths per colour group (`fast=True` or auto above a threshold);
  `VGroup` of `Dot` stays default so points remain individually animatable.
- Lines via `set_points_as_corners`, not `plot_line_graph`.
- Benchmarks in `tests/bench/`; targets: 10k-point scatter < 1 s, 100-point plot < 100 ms.

### Phase 7 — Manim-native extras
- `GGPlot.create()` ordered `AnimationGroup`; `reveal_layer(id)`.
- `transition(plot_a, plot_b, key=...)` matching marks by a key aesthetic (gganimate-style).
- `ValueTracker`-driven rebuild helper for parameterised functions/stats.

### Phase 8 — Docs & gallery
API reference; examples gallery rendered by CI; ggplot2 → ggmanim cheat sheet of supported features.
