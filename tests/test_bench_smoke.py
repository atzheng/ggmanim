"""Smoke test for benchmarks/bench.py and the ggm-bam.1 performance helpers."""

from __future__ import annotations

import importlib.util
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from manim import RED, ManimColor

from ggmanim import aes, facet_grid, facet_wrap, geom_point, ggplot
from ggmanim import build as B
from ggmanim.geoms.base import color_resolver, constant_column, num, num_array, resolve_color

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bench():
    return _load("bench")


def test_quick_scenarios_run_once(bench, tmp_path):
    out = tmp_path / "bench.json"
    assert bench.main(["--quick", "-r", "1", "--json", str(out)]) == 0
    import json

    payload = json.loads(out.read_text())
    quick = [n for n, (_, q) in bench.SCENARIOS.items() if q]
    assert sorted(payload["results"]) == sorted(quick)
    for res in payload["results"].values():
        assert res["pipeline"] > 0 and res["build"] > 0


def test_every_scenario_builds_a_plot(bench):
    for make, _ in bench.SCENARIOS.values():
        plot = make()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bench.run_pipeline(plot)


def test_compare_flags_regressions(bench):
    base = {"a": {"build": 1.0}}
    assert bench.regressions({"a": {"build": 1.4}}, base, 1.5) == []
    assert len(bench.regressions({"a": {"build": 1.6}}, base, 1.5)) == 1
    assert "vs base" in bench.table({"a": {"pipeline": 0.1, "build": 1.0}}, base)


def test_baseline_covers_every_scenario(bench):
    import json

    baseline = json.loads((ROOT / "benchmarks" / "baseline.json").read_text())["results"]
    assert set(baseline) == set(bench.SCENARIOS)


def test_compare_images(tmp_path):
    from PIL import Image

    ci = _load("compare_images")
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(a / "x.png")
    Image.new("RGB", (4, 4), (10, 20, 30)).save(b / "x.png")
    assert ci.main([str(a), str(b)]) == 0
    Image.new("RGB", (4, 4), (10, 20, 31)).save(b / "x.png")
    assert ci.main([str(a), str(b)]) == 1
    assert ci.main([str(a), str(b), "--tolerance", "1"]) == 0


# -- helpers --------------------------------------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        [0.1, None, np.nan, np.inf, 2],
        ["0.5", "abc", None, 3],
        np.array([1.0, np.nan]),
        pd.Series([0.2, None], dtype=object),
        [],
    ],
)
def test_num_array_matches_num(values):
    got = num_array(values, 0.06)
    want = [num(v, 0.06) for v in list(values)]
    assert got.dtype == float and got.tolist() == want


def test_constant_column_keeps_value_as_is():
    idx = pd.RangeIndex(3)
    col = constant_column((2.0, 1.0), idx)
    assert col.dtype == object and list(col) == [(2.0, 1.0)] * 3
    assert constant_column(None, idx).tolist() == [None, None, None]
    assert len(constant_column(0.5, pd.RangeIndex(0))) == 0


def test_color_resolver_matches_resolve_color():
    fg = ManimColor("#123456")
    resolve = color_resolver(fg)
    for v in [None, np.nan, "#ff0000", RED, "#ff0000", [1.0, 0.0, 0.0]]:
        assert resolve(v) == resolve_color(v, fg)
    assert color_resolver()(None) is None


# -- faceted pass-1 measurement reuse --------------------------------------------


def _signature(mob) -> list:
    return [
        (type(m).__name__, np.round(m.points, 9).tobytes())
        for m in mob.get_family()
        if len(m.points)
    ]


def _facet_plots():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": rng.uniform(0, 10, 120),
            "y": rng.uniform(0, 5, 120),
            "f": np.repeat(list("abcdef"), 20),
            "g": np.tile(["long strip label", "s"], 60),
        }
    )
    base = ggplot(df, aes("x", "y")).add(geom_point())
    return [
        base.add(facet_wrap("f", ncol=3)),
        base.add(facet_wrap("f", ncol=2, strip_position="left")),
        base.add(facet_wrap("f", ncol=3, scales="free_y")),
        base.add(facet_grid(rows="g", cols="f")),
    ]


@pytest.mark.parametrize("plot", _facet_plots())
def test_shared_pass1_measurement_matches_per_panel(plot, monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fast = _signature(plot.build())
        counter = itertools.count()
        monkeypatch.setattr(B, "_measure_key", lambda *a: (next(counter),))
        slow = _signature(plot.build())
    assert fast == slow
