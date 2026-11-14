# Performance

Building a plot has two parts with very different costs. The data pipeline
(layers, scales, stats, positions; steps 1–6 of the
[build](index.md#the-build-pipeline)) is pandas/numpy and takes about a
millisecond for typical plots. Making the mobjects (marks, text, legends) is
where the time goes, because manim objects are comparatively expensive to
create and copy.

## The fast path: batched marks

By default every row becomes its own mobject, which is what makes per-mark
animation possible. For large layers that is slow, so above
`theme(fast_threshold=2000)` rows per panel, `geom_point`, `geom_rect` /
`geom_tile`, `geom_col` / `geom_bar` and `geom_segment` draw **batched** marks:
one `VMobject` per distinct style (colour, fill, alpha, shape, stroke), with
every mark a subpath built straight from numpy arrays. A 20,000-point scatter
builds in well under a second.

| argument | effect |
|---|---|
| `fast=None` (default) | batch when the panel's layer has more than `theme.fast_threshold` rows and `key` is not mapped |
| `fast=True` | always batch (warns that `key` is ignored if it is mapped) |
| `fast=False` | never batch: one mobject per row |
| `theme(fast_threshold=None)` | turn the automatic switch off; `0` batches every such layer |

Batched layers look the same: marks are split into stacking layers so that
overlapping translucent marks still darken each other and draw order is kept.
What changes is identity:

- `layer.keys` is `None` and `layer.batched` is `True`;
- per-mark animation, `plot.mark()` and key-based transitions are not available
  (`plot.mark` raises `TypeError`); whole-layer reveals (`fade_in`, `create`,
  `grow_from_baseline`) still work;
- mapping `key` turns the automatic switch off, since it asks for per-mark identity.

Marks that cannot be batched (points cut by the panel edge, dashed outlines,
segments with arrows) are drawn per row and added after the batches.

`examples/large_scatter.py` builds 20,000 points and compares `fast=False` with
`fast=True` side by side:

```python
--8<-- "examples/large_scatter.py"
```

## Text

Laying out text (tick labels, legend keys, titles) is the other big cost.
`theme.make_text` memoises text mobjects: plots repeat the same strings (tick
labels shared by facet panels, legend keys, a plot rebuilt for another scene),
so each distinct request is laid out once and later requests get a fast copy.
Rebuilding a plot, or building many similar plots for a transition, is
therefore much cheaper than the first build.

## Tips

- Keep `key` unmapped on very large layers unless you need per-mark animation.
- For heatmaps on a regular grid, `geom_raster` draws one mobject per fill
  colour.
- Build once and reuse the `GGPlot`; building is pure, so a spec can also be
  built ahead of the scene.
- `benchmarks/bench.py` times the pipeline and the full build on fixed
  scenarios (`python benchmarks/bench.py --quick`), and `--profile <scenario>`
  shows where the time goes.
