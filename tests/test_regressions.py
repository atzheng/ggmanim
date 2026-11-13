"""Regression tests: one per row of the defect table in PLAN.md §1 that the new
pipeline addresses. Each test builds a plot through the public API
(``ggmanim.ggplot`` / ``GGManim`` + ``.build()``) and checks the defect is gone."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from manim import BLACK, WHITE, ManimColor, config

from ggmanim import (
    GGManim,
    aes,
    geom_point,
    ggplot,
    scale_x_log10,
    ylim,
)
from ggmanim.palettes import BRIGHT, MUTED, hue_pal


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [2.0, 4.0, 3.0, 5.0, 4.5, 6.0],
            "g": ["a", "b", "c", "a", "b", "c"],
            "s": [0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
            "w": [0.2, 0.4, 0.6, 0.8, 1.0, 0.5],
        }
    )


@pytest.fixture
def background():
    """Restore manim's background colour after a test changes it."""
    old = config.background_color
    yield
    config.background_color = old


def _hex(c) -> str:
    return ManimColor(c).to_hex().upper()


# -- add_layer: `data or self.data` raised on per-layer DataFrames ----------------


def test_per_layer_dataframe(df):
    other = pd.DataFrame({"x": [10.0, 20.0], "y": [0.0, 1.0]})
    m = ggplot(df, aes("x", "y")).geom_point().geom_line(data=other).build()
    assert len(m.layers[0]) == len(df)
    assert m.layer_data[1]["x"].tolist() == [10.0, 20.0]
    # both layers train the shared scales
    assert m.scales["x"].get_limits() == (1.0, 20.0)
    assert m.scales["y"].get_limits() == (0.0, 6.0)


def test_per_layer_dataframe_without_plot_data():
    a = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    b = pd.DataFrame({"x": [2.0, 3.0], "y": [5.0, 4.0]})
    m = ggplot(mapping=aes("x", "y")).add(geom_point(data=a), geom_point(data=b)).build()
    assert m.scales["x"].get_limits() == (0.0, 3.0)


# -- build: layers without x/y aes (function/vline/hline) broke auto limits -------


def test_intercept_and_function_layers_with_auto_limits(df):
    m = (
        ggplot(df, aes("x", "y"))
        .geom_point()
        .geom_function(lambda x: 0.5 * x)
        .geom_vline(xintercept=7)
        .geom_hline(yintercept=0)
        .build()
    )
    assert m.scales["x"].get_limits() == (1.0, 7.0)
    assert m.scales["y"].get_limits() == (0.0, 6.0)
    assert len(m.layers) == 4


def test_intercept_only_plot_builds():
    m = ggplot().geom_vline(xintercept=3).geom_hline(yintercept=[1, 2]).build()
    lo, hi = m.panel_params.x_range
    assert lo < 3 < hi  # zero-width range is widened, not degenerate
    assert m.scales["y"].get_limits() == (1.0, 2.0)


# -- auto_breaks: 2-16 ticks, no 1-2-5 rounding -----------------------------------


@pytest.mark.parametrize(
    "lo,hi",
    [
        (0, 1),
        (0, 7),
        (0, 10),
        (0, 13),
        (1, 99),
        (0.001, 0.0123),
        (-3.2, 47),
        (100, 1000),
        (12345, 12399),
        (-1e6, 3e6),
        (0.5, 0.51),
        (2, 5.5),
    ],
)
def test_nice_break_count(lo, hi):
    data = pd.DataFrame({"x": [lo, hi], "y": [lo, hi]})
    m = ggplot(data, aes("x", "y")).geom_point().build()
    for axis in ("x", "y"):
        breaks = np.asarray(getattr(m.panel_params, f"{axis}_breaks"))
        rng = getattr(m.panel_params, f"{axis}_range")
        assert 4 <= len(breaks) <= 8, (lo, hi, breaks)
        assert np.all((breaks >= rng[0] - 1e-9) & (breaks <= rng[1] + 1e-9))
        # equally spaced with a 1-2-5 step
        steps = np.diff(breaks)
        np.testing.assert_allclose(steps, steps[0], rtol=1e-6)
        mantissa = steps[0] / 10 ** np.floor(np.log10(steps[0]))
        assert any(np.isclose(mantissa, v) for v in (1, 2, 2.5, 5)), steps[0]


# -- scale_*_log10 + LogBase: range passed in data units --------------------------


def test_log10_axis_labels():
    data = pd.DataFrame({"x": [1.0, 10.0, 100.0, 1000.0], "y": [1.0, 2.0, 3.0, 4.0]})
    m = ggplot(data, aes("x", "y")).geom_point().add(scale_x_log10()).build()
    pp = m.panel_params
    # trained, broken and drawn in log space
    assert m.scales["x"].get_limits() == pytest.approx((0.0, 3.0))
    assert pp.x_range[0] < 0.0 and pp.x_range[1] > 3.0
    # labels are data values at the (log-space) break positions
    for b, lab in zip(pp.x_breaks, pp.x_labels):
        assert float(lab) == pytest.approx(10**b)
    assert {"1", "10", "100", "1000"} <= set(pp.x_labels)
    # points sit at c2p(log10(x))
    centres = np.array([d.get_center() for d in m.layers[0]])
    np.testing.assert_allclose(
        centres, m.panel.transform(np.log10(data["x"]), data["y"]), atol=1e-6
    )


# -- colour scale: only `color` mapped; others leaked raw values into kwargs -----


def test_fill_mapping_uses_palette(df):
    m = ggplot(df, aes("x", "y", fill="g")).geom_col().build()
    fills = {_hex(b.get_fill_color()) for b in m.layers[0]}
    assert fills == {_hex(c) for c in m.scales["fill"].get_palette_values()}
    assert len(fills) == 3 and len(m.legends) == 1


@pytest.mark.parametrize("geom", ["geom_point", "geom_line", "geom_col", "geom_smooth"])
@pytest.mark.parametrize(
    "mapping",
    [
        dict(fill="g"),
        dict(shape="g"),
        dict(linetype="g"),
        dict(size="g"),
        dict(alpha="g"),
        dict(size="s"),
        dict(alpha="w"),
    ],
)
def test_mapped_non_colour_aesthetics_build(df, geom, mapping):
    # used to raise TypeError: raw column values passed as mobject kwargs
    m = getattr(ggplot(df, aes("x", "y", **mapping)), geom)().build()
    for aes_name in mapping:
        if aes_name in ("fill", "shape", "linetype"):
            assert aes_name in m.scales
            assert not set(m.layer_data[0][aes_name]) & set(df["g"])  # mapped, not raw


def test_numeric_size_and_alpha_reach_dots(df):
    m = ggplot(df, aes("x", "y", size="s", alpha="w")).geom_point().build()
    dots = m.layers[0]
    np.testing.assert_allclose([d.radius for d in dots], df["s"])
    np.testing.assert_allclose([d.get_fill_opacity() for d in dots], df["w"])


# -- palettes.HashableManimColor: palettes must be plain sequences ----------------


def test_palettes_are_plain_lists():
    for pal in (BRIGHT, MUTED, hue_pal()(4)):
        assert isinstance(pal, list)
        arr = np.asarray(pal, dtype=object)
        assert arr.shape == (len(pal),)  # colours are not expanded element-wise


# -- build_geom_line: label used the last group only; KeyError grouping by fill --


def test_grouped_line_labels_per_group(df):
    m = ggplot(df, aes("x", "y", color="g")).geom_line(label=["A", "B", "C"]).build()
    layer = m.layers[0]
    assert len(layer.marks) == 3 and len(layer.labels) == 3
    assert [t.text for t in layer.labels] == ["A", "B", "C"]
    for path, lab in zip(layer.marks, layer.labels):
        assert _hex(lab.get_color()) == _hex(path.get_stroke_color())
        # each label sits right of its own group's last point
        assert lab.get_left()[0] > path.get_end()[0]
        assert abs(lab.get_center()[1] - path.get_end()[1]) < 0.3


def test_line_grouped_by_fill_with_labels(df):
    m = ggplot(df, aes("x", "y", fill="g")).geom_line(label="z").build()
    assert len(m.layers[0].marks) == 3 and len(m.layers[0].labels) == 3


# -- build_geom_smooth: linear only, no SE ribbon, no method= ---------------------


def test_smooth_lm_with_se_ribbon_and_degree(df):
    m = ggplot(df, aes("x", "y")).geom_smooth(method="lm").build()
    d = m.layer_data[0]
    slope, intercept = np.polyfit(df["x"], df["y"], 1)
    np.testing.assert_allclose(d["y"], slope * d["x"] + intercept, atol=1e-9)
    assert (d["ymin"] < d["y"]).all() and (d["y"] < d["ymax"]).all()
    m2 = ggplot(df, aes("x", "y")).geom_smooth(method="lm", se=False).build()
    assert "ymin" not in m2.layer_data[0] or m2.layer_data[0]["ymin"].isna().all()
    with pytest.raises(ValueError, match="no-such-method"):
        ggplot(df, aes("x", "y")).geom_smooth(method="no-such-method").build()


# -- build_geom_bar: bars anchored at y=0 even when out of range; log-x widths ---


def test_bars_clipped_to_panel(df):
    m = ggplot(df, aes("x", "y")).geom_col().add(ylim(2, 7)).build()
    y_lo = m.panel.transform(0.0, m.panel_params.y_range[0])[0, 1]
    for bar in m.layers[0]:
        assert bar.get_bottom()[1] >= y_lo - 1e-6


def test_bars_equal_width_on_log_x():
    data = pd.DataFrame({"x": [1.0, 10.0, 100.0], "y": [1.0, 2.0, 3.0]})
    m = ggplot(data, aes("x", "y")).geom_col().add(scale_x_log10()).build()
    widths = [b.width for b in m.layers[0]]
    np.testing.assert_allclose(widths, widths[0], rtol=1e-6)


# -- build: y-label rotated in place -> second build() rotated it 180 degrees -----


def test_double_build_idempotent(df):
    p = ggplot(df, aes("x", "y", color="g")).geom_point().geom_line().labs(title="T", y="response")
    a, b = p.build(), p.build()
    for attr in ("x_title", "y_title"):
        ta, tb = getattr(a.panel, attr), getattr(b.panel, attr)
        np.testing.assert_allclose(ta.points, tb.points, atol=1e-9)
    # the y title is vertical (rotated once), not back to horizontal
    assert a.panel.y_title.height > a.panel.y_title.width
    np.testing.assert_allclose(a.get_all_points(), b.get_all_points(), atol=1e-9)
    assert len(p.layers) == 2  # the spec is not mutated


def test_legacy_gGManim_alias_chaining(df):
    m = GGManim(df, aes("x", "y")).geom_point(layer_id="pts").xlab("X").ylab("Y").build()
    assert m["layers"]["pts"] is m.layers["pts"]
    assert m.panel.x_title.text == "X"


# -- Axes(...).set_color(BLACK): hard-coded for a white background ----------------


@pytest.mark.parametrize("bg,fg", [(BLACK, WHITE), (WHITE, BLACK), ("#1e1e1e", WHITE)])
def test_foreground_follows_background(df, background, bg, fg):
    config.background_color = bg
    m = ggplot(df, aes("x", "y")).geom_point().build()
    assert _hex(m.axes.x_axis.get_color()) == _hex(fg)
    assert _hex(m.panel.x_title.get_color()) == _hex(fg)
    assert _hex(m.layers[0][0].get_fill_color()) == _hex(fg)


# -- __init__.py: `import *` without __all__ re-exported f, np, pd, ... ----------


def test_star_import_exposes_exactly_all():
    import ggmanim

    ns: dict = {}
    exec("from ggmanim import *", ns)
    ns.pop("__builtins__", None)
    assert set(ns) == set(ggmanim.__all__)
    assert len(ggmanim.__all__) == len(set(ggmanim.__all__))
    for leaked in ("np", "pd", "f", "manim", "LinearRegression", "Dot", "VGroup"):
        assert leaked not in ns


def test_every_registered_component_is_public():
    # in a fresh interpreter: other tests register throwaway components
    code = (
        "import ggmanim, ggmanim.registry as r; "
        "missing = sorted(set(r.COMPONENTS) - set(ggmanim.__all__)); "
        "assert not missing, missing; print(len(r.COMPONENTS))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert int(out.stdout) >= 30


def test_public_api_names():
    import ggmanim

    for name in ("ggplot", "GGManim", "aes", "after_stat", "theme", "labs", "xlab", "ylab"):
        assert name in ggmanim.__all__
    for name in ("xlim", "ylim", "guides", "coord_cartesian", "hue_pal", "BRIGHT", "PALETTES"):
        assert name in ggmanim.__all__
    assert ggmanim.GGManim is ggmanim.ggplot


# -- setup.py: distutils, no install_requires; funcy/sklearn undeclared ----------


ROOT = Path(__file__).resolve().parents[1]


def test_packaging_metadata():
    tomllib = pytest.importorskip("tomllib")
    assert not (ROOT / "setup.py").exists()
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert meta["build-system"]["build-backend"] == "setuptools.build_meta"
    project = meta["project"]
    assert project["name"] == "ggmanim"
    assert project["requires-python"] == ">=3.10"
    deps = " ".join(project["dependencies"])
    assert "manim>=0.19" in deps and "pandas>=2" in deps and "numpy" in deps
    assert project["optional-dependencies"]["stats"] == ["scipy"]
    everything = deps + " ".join(sum(project["optional-dependencies"].values(), []))
    assert "funcy" not in everything and "scikit-learn" not in everything


# -- repo: legacy prototype, stray gurobi.log; funcy/sklearn imports ------------


def test_legacy_code_removed():
    assert not (ROOT / "ggmanim" / "GGManim.py").exists()
    for path in (ROOT / "ggmanim").rglob("*.py"):
        src = path.read_text()
        for needle in ("funcy", "sklearn", "HashableManimColor", "distutils"):
            assert needle not in src, (path, needle)
    import ggmanim

    assert not hasattr(ggmanim, "CATEGORICAL_AES")
    assert not hasattr(ggmanim, "color_mapping")
