# Benchmarks

`bench.py` times the build pipeline on fixed scenarios with plain `timeit`
(no extra dependencies). Each timing is the best of `--repeat` runs after a
warm-up run.

| scenario | what |
|---|---|
| `scatter_100`, `scatter_1k`, `scatter_10k` | `geom_point`, n random points |
| `lines_1k_grouped` | `geom_line`, 10 coloured groups of 100 points |
| `col_500` | `geom_col`, 500 bars |
| `tile_50x50` | `geom_tile`, 2500 tiles, continuous fill |
| `facet_3x3` | `facet_wrap` into 3 x 3 panels, points + `lm` smooth |
| `legend_heavy` | 600 points with colour (12 levels), shape and size legends |

Two timings per scenario:

* **pipeline**: the data stages only (`ggmanim.build` steps 1-6: layers,
  panels, scales, stats, positions); no manim objects.
* **build**: the whole `build_plot` (panels, marks, legends, layout).

```bash
python benchmarks/bench.py                    # all scenarios
python benchmarks/bench.py --quick            # the small ones (also what the smoke test runs)
python benchmarks/bench.py -k scatter_10k -r 5
python benchmarks/bench.py --json out.json --compare benchmarks/baseline.json --threshold 1.5
python benchmarks/bench.py --profile scatter_10k   # cProfile top 30 + manim / non-manim split
```

`--compare` adds a "vs base" speed-up column and prints a `::warning::` line
for every timing slower than `threshold` x the baseline; `--strict` makes that
fail (exit 1). The `bench` GitHub workflow runs the full set, uploads the JSON
and compares it with `baseline.json` without `--strict`, since CI machines
differ from the machine that wrote the baseline.

`baseline.json` was written on an Apple-silicon Mac (Python 3.11, manim 0.22,
pandas 3.1). Regenerate it after an intentional performance change:

```bash
python benchmarks/bench.py -r 5 --json benchmarks/baseline.json
```

## Image comparison

`compare_images.py` checks that a change leaves the rendered gallery
pixel-identical:

```bash
MANIM=manim bash scripts/render_examples.sh /tmp/before   # on the old commit
MANIM=manim bash scripts/render_examples.sh /tmp/after    # on the new one
python benchmarks/compare_images.py /tmp/before/images /tmp/after/images
```

## Results of ggm-bam.1

Phase 2 baseline vs after ggm-bam.1, same machine, best of 5 (seconds):

| scenario | pipeline before | pipeline after | build before | build after | build speed-up |
|---|---:|---:|---:|---:|---:|
| scatter_100 | 0.0011 | 0.0011 | 0.060 | 0.058 | 1.0x |
| scatter_1k | 0.0011 | 0.0010 | 0.143 | 0.118 | 1.2x |
| scatter_10k | 0.0012 | 0.0012 | 1.472 | 0.731 | 2.0x |
| lines_1k_grouped | 0.0023 | 0.0022 | 0.156 | 0.152 | 1.0x |
| col_500 | 0.125 | 0.122 | 0.252 | 0.246 | 1.0x |
| tile_50x50 | 0.011 | 0.011 | 0.477 | 0.470 | 1.0x |
| facet_3x3 | 0.034 | 0.035 | 0.788 | 0.621 | 1.3x |
| legend_heavy | 0.0033 | 0.0031 | 0.342 | 0.327 | 1.0x |

See `PROFILE.md` for the 10k-scatter profile. Known remaining hot spots
(outside this bead's files): `position_stack` on `col_500` (~0.12 s of the
pipeline), per-value `ManimColor` construction in continuous colour palettes
(`tile_50x50`, ~9 ms), `make_marker`'s per-marker `deepcopy` and text
rendering (`theme.make_text`, ggm-bam.2).
