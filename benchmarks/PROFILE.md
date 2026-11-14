# Profile: 10k-point scatter (`scatter_10k`)

Generated with `python benchmarks/bench.py --profile scatter_10k` (cProfile of
one `build_plot` after a warm-up; Python 3.11, manim 0.22, pandas 3.1, Apple
silicon). Profiling roughly doubles every time; use `bench.py` for timings.

## Summary (ggm-bam.1)

| | Phase 2 baseline | after ggm-bam.1 | speed-up |
|---|---:|---:|---:|
| `build` (bench, best of 5) | 1.496 s | 0.742 s | 2.0x |
| profiled self time outside manim (ggmanim, pandas, numpy) | 0.384 s | 0.190 s | 2.0x |
| profiled self time inside manim (incl. `copy.deepcopy`) | 2.828 s | 2.061 s | 1.4x |
| data pipeline, steps 1-6 (bench) | 1.3 ms | 1.2 ms | ~1x |

What changed for this scenario:

* `VGroup.add` per dot was quadratic: manim's `_insert_submobjects` runs
  `list.index` over the existing children on every single add (0.59 s of
  `{method 'index' of 'list' objects}` for 10k dots below). `GeomPoint` now
  collects the markers and builds one `VGroup(*marks)`.
* `GeomPoint.draw_panel` no longer iterates `DataFrame.to_dict("records")`:
  sizes and alphas are converted once as arrays (`num_array`), colours are
  resolved once per distinct value (`color_resolver`), the other columns are
  read with `tolist()`.
* Markers whose centre is further than `marker_extent` from every panel edge
  are surely unclipped and skip `clip_marker` (which returns them unchanged).
* The data pipeline (steps 1-6) costs ~1 ms here, dominated by fixed pandas
  per-call overhead; its deep copies became shallow (whole columns are
  replaced, never written into) and `map_non_position` copies once per layer.

What is left is manim: `make_marker` deep-copies a prototype per marker
(`Mobject.copy` -> `copy.deepcopy`, ~1.2 s profiled), colour arrays
(`update_rgbas_array`) and `fit_to_frame` walking 10k families. Batched marks
(ggm-bam.3) are the way past that.

The "outside manim" split shares the self time of builtins out to their
callers (see `profile_split` in `bench.py`), so the quadratic `list.index`
counts as manim even though ggmanim caused it.

## After (top 30 by cumulative time)

```
         5974689 function calls (5401938 primitive calls) in 2.250 seconds

   Ordered by: cumulative time
   List reduced from 1037 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.010    0.010    2.250    2.250 bench.py:167(run_build)
        1    0.000    0.000    2.241    2.241 build.py:910(build_plot)
        1    0.000    0.000    1.838    1.838 build.py:764(draw_layers)
        1    0.020    0.020    1.837    1.837 point.py:43(draw_panel)
    10000    0.040    0.000    1.755    0.000 _markers.py:83(make_marker)
    10000    0.007    0.000    1.167    0.000 mobject.py:1025(copy)
480000/10000    0.530    0.000    1.159    0.000 copy.py:128(deepcopy)
    10000    0.177    0.000    1.132    0.000 mobject.py:458(__deepcopy__)
    21064    0.048    0.000    0.331    0.000 vectorized_mobject.py:256(update_rgbas_array)
        1    0.000    0.000    0.310    0.310 build.py:844(assemble)
        1    0.000    0.000    0.310    0.310 build.py:890(fit_to_frame)
    21064    0.061    0.000    0.271    0.000 vectorized_mobject.py:223(generate_rgbas_array)
    30000    0.049    0.000    0.222    0.000 copy.py:259(_reconstruct)
10411/10397    0.007    0.000    0.190    0.000 vectorized_mobject.py:282(set_fill)
        1    0.000    0.000    0.178    0.178 mobject.py:2033(move_to)
10653/10625    0.011    0.000    0.162    0.000 vectorized_mobject.py:332(set_stroke)
       74    0.000    0.000    0.139    0.002 mobject.py:2395(get_critical_point)
       76    0.008    0.000    0.136    0.002 vectorized_mobject.py:1791(get_points_defining_boundary)
       13    0.000    0.000    0.131    0.010 mobject.py:2491(length_over_dim)
 40302/26    0.053    0.000    0.131    0.005 mobject.py:2297(reduce_across_dimension)
    20085    0.014    0.000    0.123    0.000 mobject.py:2711(family_members_with_points)
    10044    0.037    0.000    0.121    0.000 mobject.py:1391(shift)
    10013    0.006    0.000    0.116    0.000 vectorized_mobject.py:484(scale)
    10013    0.006    0.000    0.109    0.000 mobject.py:1416(scale)
    10041    0.040    0.000    0.105    0.000 mobject.py:1698(apply_points_function_about_point)
    20175    0.024    0.000    0.102    0.000 mobject.py:2667(get_family)
        1    0.000    0.000    0.088    0.088 build.py:618(build_panel)
        1    0.000    0.000    0.088    0.088 coords.py:316(build)
    10322    0.004    0.000    0.088    0.000 vectorized_mobject.py:1794(<genexpr>)
   150000    0.063    0.000    0.085    0.000 copy.py:243(_keep_alive)
```

## Phase 2 baseline (top 30 by cumulative time)

```
         6576781 function calls (6004002 primitive calls) in 3.212 seconds

   Ordered by: cumulative time
   List reduced from 1082 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.010    0.010    3.212    3.212 bench.py:167(run_build)
        1    0.000    0.000    3.202    3.202 build.py:881(build_plot)
        1    0.000    0.000    2.837    2.837 build.py:735(draw_layers)
        1    0.022    0.022    2.836    2.836 point.py:40(draw_panel)
    10000    0.023    0.000    1.878    0.000 point.py:64(_dot)
    10000    0.043    0.000    1.822    0.000 _markers.py:83(make_marker)
    10000    0.007    0.000    1.220    0.000 mobject.py:1025(copy)
480000/10000    0.567    0.000    1.212    0.000 copy.py:128(deepcopy)
    10000    0.181    0.000    1.183    0.000 mobject.py:458(__deepcopy__)
    10260    0.022    0.000    0.706    0.000 vectorized_mobject.py:2253(add)
    10271    0.010    0.000    0.681    0.000 mobject.py:568(add)
    10271    0.035    0.000    0.670    0.000 mobject.py:491(_insert_submobjects)
    10045    0.588    0.000    0.591    0.000 {method 'index' of 'list' objects}
    21064    0.049    0.000    0.338    0.000 vectorized_mobject.py:256(update_rgbas_array)
    21064    0.063    0.000    0.277    0.000 vectorized_mobject.py:223(generate_rgbas_array)
        1    0.000    0.000    0.273    0.273 build.py:815(assemble)
        1    0.000    0.000    0.272    0.272 build.py:861(fit_to_frame)
    30000    0.050    0.000    0.257    0.000 copy.py:259(_reconstruct)
10411/10397    0.007    0.000    0.196    0.000 vectorized_mobject.py:282(set_fill)
    30085    0.022    0.000    0.168    0.000 mobject.py:2711(family_members_with_points)
10653/10625    0.011    0.000    0.163    0.000 vectorized_mobject.py:332(set_stroke)
    10000    0.043    0.000    0.143    0.000 _marker_clip.py:116(clip_marker)
        1    0.000    0.000    0.137    0.137 mobject.py:2033(move_to)
       13    0.000    0.000    0.135    0.010 mobject.py:2491(length_over_dim)
 40302/26    0.056    0.000    0.135    0.005 mobject.py:2297(reduce_across_dimension)
    30175    0.034    0.000    0.131    0.000 mobject.py:2667(get_family)
    10044    0.038    0.000    0.120    0.000 mobject.py:1391(shift)
    10013    0.007    0.000    0.118    0.000 vectorized_mobject.py:484(scale)
    10013    0.006    0.000    0.111    0.000 mobject.py:1416(scale)
    10041    0.039    0.000    0.106    0.000 mobject.py:1698(apply_points_function_about_point)
```
