"""Benchmark harness for the ggmanim build pipeline (plain ``timeit``, no extra deps).

Each scenario builds a plot spec once (outside the timing) and times two things:

``pipeline``
    The data stages only -- resolve layers, assign panels, position scales,
    stats, non-position scales, ``setup_data``, positions and the final scale
    training (:mod:`ggmanim.build` steps 1-6). No manim objects are made.
``build``
    The whole :func:`ggmanim.build.build_plot` (panel, marks, legends, layout).

Usage::

    python benchmarks/bench.py                       # every scenario, table on stdout
    python benchmarks/bench.py --quick               # the small scenarios only
    python benchmarks/bench.py -k scatter_10k -r 5   # one scenario, best of 5
    python benchmarks/bench.py --json out.json       # also write the results as JSON
    python benchmarks/bench.py --compare benchmarks/baseline.json --threshold 1.5

``--compare`` prints the ratio to a committed baseline and warns (``::warning::``
lines in GitHub Actions) for every timing slower than ``threshold`` x the baseline;
``--strict`` turns those warnings into exit status 1. Timings are the best of
``--repeat`` runs (after one warm-up run).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import timeit
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # run from a checkout without installing
    sys.path.insert(0, str(ROOT))

import ggmanim  # noqa: E402
from ggmanim import (  # noqa: E402
    aes,
    facet_wrap,
    geom_col,
    geom_line,
    geom_point,
    geom_smooth,
    geom_tile,
    ggplot,
)
from ggmanim import build as B  # noqa: E402

# --------------------------------------------------------------------- scenarios


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def scatter(n: int) -> Callable[[], ggplot]:
    def make() -> ggplot:
        rng = _rng(1)
        df = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})
        return ggplot(df, aes("x", "y")).add(geom_point(size=0.02))

    return make


def grouped_lines() -> ggplot:
    """1000 rows: 10 coloured lines of 100 points."""
    rng = _rng(2)
    g = np.repeat([f"g{i}" for i in range(10)], 100)
    x = np.tile(np.arange(100.0), 10)
    y = rng.normal(size=1000).cumsum()
    df = pd.DataFrame({"x": x, "y": y, "g": g})
    return ggplot(df, aes("x", "y", color="g")).add(geom_line())


def bars_500() -> ggplot:
    rng = _rng(3)
    df = pd.DataFrame({"x": np.arange(500.0), "y": rng.uniform(1, 10, 500)})
    return ggplot(df, aes("x", "y")).add(geom_col())


def bars_stacked() -> ggplot:
    """500 x positions x 4 stacked groups (2000 bars, ``position_stack``)."""
    rng = _rng(6)
    df = pd.DataFrame(
        {
            "x": np.repeat(np.arange(500.0), 4),
            "y": rng.uniform(1, 10, 2000),
            "g": np.tile(list("abcd"), 500),
        }
    )
    return ggplot(df, aes("x", "y", fill="g")).add(geom_col())


def scatter_markers(n: int) -> Callable[[], ggplot]:
    """``geom_point(fast=False)``: one ``make_marker`` per row."""

    def make() -> ggplot:
        rng = _rng(1)
        df = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})
        return ggplot(df, aes("x", "y")).add(geom_point(size=0.02, fast=False))

    return make


def tile(n: int) -> Callable[[], ggplot]:
    def make() -> ggplot:
        xx, yy = np.meshgrid(np.arange(float(n)), np.arange(float(n)))
        df = pd.DataFrame({"x": xx.ravel(), "y": yy.ravel()})
        df["z"] = np.sin(df["x"] / 5) * np.cos(df["y"] / 7)
        return ggplot(df, aes("x", "y", fill="z")).add(geom_tile())

    return make


def tile_50x50() -> ggplot:
    xx, yy = np.meshgrid(np.arange(50.0), np.arange(50.0))
    df = pd.DataFrame({"x": xx.ravel(), "y": yy.ravel()})
    df["z"] = np.sin(df["x"] / 5) * np.cos(df["y"] / 7)
    return ggplot(df, aes("x", "y", fill="z")).add(geom_tile())


def facet_3x3() -> ggplot:
    rng = _rng(4)
    n = 900
    df = pd.DataFrame(
        {
            "x": rng.uniform(0, 10, n),
            "panel": np.repeat([f"p{i}" for i in range(9)], n // 9),
        }
    )
    df["y"] = 2 * df["x"] + rng.normal(0, 2, n)
    return (
        ggplot(df, aes("x", "y"))
        .add(geom_point(size=0.03))
        .add(geom_smooth(method="lm", se=False))
        .add(facet_wrap("panel", ncol=3))
    )


def legend_heavy() -> ggplot:
    """Colour, shape and size legends (12 colour levels, 6 shapes)."""
    rng = _rng(5)
    n = 600
    df = pd.DataFrame(
        {
            "x": rng.uniform(0, 10, n),
            "y": rng.uniform(0, 10, n),
            "c": rng.choice([f"level {i:02d}" for i in range(12)], n),
            "s": rng.choice(list("abcdef"), n),
            "z": rng.uniform(1, 100, n),
        }
    )
    return ggplot(df, aes("x", "y", color="c", shape="s", size="z")).add(geom_point(alpha=0.7))


#: name -> (plot factory, part of ``--quick``)
SCENARIOS: dict[str, tuple[Callable[[], ggplot], bool]] = {
    "scatter_100": (scatter(100), True),
    "scatter_1k": (scatter(1_000), True),
    "scatter_10k": (scatter(10_000), False),
    "lines_1k_grouped": (grouped_lines, True),
    "col_500": (bars_500, False),
    "tile_50x50": (tile_50x50, False),
    "col_stacked_2k": (bars_stacked, False),
    "scatter_markers_2k": (scatter_markers(2_000), False),
    "tile_100x100": (tile(100), False),
    "facet_3x3": (facet_3x3, False),
    "legend_heavy": (legend_heavy, False),
}


# --------------------------------------------------------------------- timings


def run_pipeline(plot: ggplot) -> None:
    """Steps 1-6 of :func:`ggmanim.build.build_plot` (no manim objects)."""
    layers = B.resolve_layers(plot)
    layout = B.assign_panels(layers, plot)
    if not B._facet(plot).is_null:
        panel_scales = B.setup_panel_scales(plot, layers, layout)
        first = panel_scales[min(panel_scales)]
        B.compute_stats(layers, first, panel_scales, layout)
        B.map_non_position(plot, layers)
        B.setup_geom_data(layers, first)
        B.apply_positions(layers, first, panel_scales)
        B.train_position_scales(layers, first, panel_scales)
        return
    scales = B.setup_position_scales(plot, layers)
    B.compute_stats(layers, scales)
    B.map_non_position(plot, layers)
    B.setup_geom_data(layers, scales)
    B.apply_positions(layers, scales)
    B.train_position_scales(layers, scales)


def run_build(plot: ggplot) -> None:
    B.build_plot(plot)


def best_of(fn: Callable[[], object], repeat: int) -> float:
    fn()  # warm-up (imports, caches)
    return min(timeit.repeat(fn, number=1, repeat=repeat))


def bench(names: list[str], repeat: int, build: bool = True) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in names:
            plot = SCENARIOS[name][0]()
            res = {"pipeline": best_of(lambda p=plot: run_pipeline(p), repeat)}
            if build:
                res["build"] = best_of(lambda p=plot: run_build(p), repeat)
            out[name] = res
            print(f"  {name:<18} " + "  ".join(f"{k} {v:8.4f}s" for k, v in res.items()))
    return out


# --------------------------------------------------------------------- profile

#: path fragments of the code that counts as "manim" in :func:`profile_split`
MANIM_PARTS = ("/manim/", "/svgelements/", "/copy.py", "/manimpango/", "/cairo/")


def profile_split(name: str, top: int = 30) -> tuple[str, dict[str, float]]:
    """cProfile one ``build`` of scenario ``name``.

    Returns the ``pstats`` report (``top`` functions by cumulative time) and the
    split of the profiled time into ``manim`` (self time of manim / svgelements /
    ``copy`` functions, plus builtins they call) and ``non_manim`` (the rest:
    ggmanim, pandas, numpy and the builtins they call). Profiling inflates
    every number; compare the split between versions, not with ``bench``.
    """
    import cProfile
    import io
    import pstats

    plot = SCENARIOS[name][0]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_build(plot)  # warm-up
        prof = cProfile.Profile()
        prof.enable()
        run_build(plot)
        prof.disable()
    buf = io.StringIO()
    stats = pstats.Stats(prof, stream=buf)
    stats.strip_dirs().sort_stats("cumulative").print_stats(top)

    raw = pstats.Stats(prof).stats  # type: ignore[attr-defined]

    def is_manim(func) -> bool:
        return any(part in func[0] for part in MANIM_PARTS)

    split = {"manim": 0.0, "non_manim": 0.0}
    for func, (_cc, _nc, tt, _ct, callers) in raw.items():
        if func[0] != "~":  # a Python function: by its own file
            split["manim" if is_manim(func) else "non_manim"] += tt
            continue
        # a builtin: share its self time out by the time each caller spent calling it
        total = sum(c[3] for c in callers.values()) or 1.0
        for caller, (_c1, _c2, _t, ct) in callers.items():
            split["manim" if is_manim(caller) else "non_manim"] += tt * ct / total
    split["total"] = split["manim"] + split["non_manim"]
    return buf.getvalue(), split


# --------------------------------------------------------------------- report


def table(results: dict, baseline: dict | None = None) -> str:
    cols = ["pipeline", "build"]
    head = f"{'scenario':<18}" + "".join(f"{c + ' (s)':>15}" for c in cols)
    if baseline:
        head += "".join(f"{c + ' vs base':>18}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, res in results.items():
        row = f"{name:<18}" + "".join(
            f"{res[c]:>15.4f}" if c in res else f"{'-':>15}" for c in cols
        )
        if baseline:
            base = baseline.get(name, {})
            for c in cols:
                if c in res and base.get(c):
                    row += f"{base[c] / res[c]:>17.2f}x"
                else:
                    row += f"{'-':>18}"
        lines.append(row)
    return "\n".join(lines)


def regressions(results: dict, baseline: dict, threshold: float) -> list[str]:
    msgs = []
    for name, res in results.items():
        for c, t in res.items():
            b = baseline.get(name, {}).get(c)
            if b and t > threshold * b:
                msgs.append(f"{name} {c}: {t:.4f}s vs baseline {b:.4f}s ({t / b:.2f}x slower)")
    return msgs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-k", "--scenario", action="append", help="scenario name (repeatable)")
    ap.add_argument("--quick", action="store_true", help="only the small scenarios")
    ap.add_argument("-r", "--repeat", type=int, default=3, help="runs per timing (best of)")
    ap.add_argument("--no-build", action="store_true", help="time the data pipeline only")
    ap.add_argument("--json", type=Path, help="write results to this JSON file")
    ap.add_argument("--compare", type=Path, help="baseline JSON to compare against")
    ap.add_argument("--threshold", type=float, default=1.5, help="slow-down that warns")
    ap.add_argument("--strict", action="store_true", help="exit 1 on a regression")
    ap.add_argument("--list", action="store_true", help="list the scenarios and exit")
    ap.add_argument(
        "--profile", metavar="SCENARIO", help="cProfile one build: top 30 + manim/non-manim split"
    )
    args = ap.parse_args(argv)

    if args.profile is not None:
        report, split = profile_split(args.profile)
        print(report)
        print("profiled self time (s): " + ", ".join(f"{k} {v:.3f}" for k, v in split.items()))
        return 0

    if args.list:
        for name, (_, quick) in SCENARIOS.items():
            print(name + ("  (quick)" if quick else ""))
        return 0
    names = args.scenario or [n for n, (_, q) in SCENARIOS.items() if q or not args.quick]
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        ap.error(f"unknown scenario(s) {unknown}; see --list")

    print(f"ggmanim benchmarks (best of {args.repeat}):")
    results = bench(names, args.repeat, build=not args.no_build)
    baseline = None
    if args.compare is not None:
        baseline = json.loads(args.compare.read_text())["results"]
    print()
    print(table(results, baseline))

    if args.json is not None:
        payload = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "ggmanim": getattr(ggmanim, "__version__", "dev"),
            "repeat": args.repeat,
            "results": results,
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")

    if baseline is not None:
        slow = regressions(results, baseline, args.threshold)
        for msg in slow:
            print(f"::warning::benchmark regression: {msg}")
        if slow and args.strict:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
