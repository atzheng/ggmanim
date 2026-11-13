import math

import numpy as np
import pandas as pd
import pytest

from ggmanim.stats import (
    NO_GROUP,
    Stat,
    StatSmooth,
    add_group,
    get_stat,
    stat_function,
    stat_identity,
    stat_smooth,
)
from ggmanim.stats import smooth as smooth_mod


class FakeScale:
    def __init__(self, limits, trans=None):
        self._limits = limits
        self.trans = trans

    def get_limits(self):
        return self._limits


class Log10:
    @staticmethod
    def transform(x):
        return np.log10(x)

    @staticmethod
    def inverse(x):
        return 10.0 ** np.asarray(x)


# -- add_group ---------------------------------------------------------------

def test_add_group_no_grouping_aes():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
    out = add_group(df)
    assert (out["group"] == NO_GROUP).all()
    assert "group" not in df.columns  # input untouched


def test_add_group_discrete_aes_interaction():
    df = pd.DataFrame({
        "x": range(6),
        "color": ["b", "a", "b", "a", "b", "a"],
        "shape": ["u", "u", "v", "v", "u", "u"],
    })
    out = add_group(df)
    # sorted interaction: (a,u)=1, (a,v)=2, (b,u)=3, (b,v)=4
    assert out["group"].tolist() == [3, 1, 4, 2, 3, 1]


def test_add_group_ignores_continuous_color():
    df = pd.DataFrame({"x": [1, 2], "color": [0.1, 0.2]})
    assert (add_group(df)["group"] == NO_GROUP).all()


def test_add_group_existing_group_column():
    df = pd.DataFrame({"x": [1, 2, 3], "group": ["z", "y", "z"]})
    assert add_group(df)["group"].tolist() == [2, 1, 2]
    df = pd.DataFrame({"x": [1, 2], "group": [7, 9]})
    assert add_group(df)["group"].tolist() == [7, 9]


# -- stat_identity -----------------------------------------------------------

def test_identity_passthrough():
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "color": ["a", "b"]})
    out = stat_identity().compute(df)
    pd.testing.assert_frame_equal(out[["x", "y", "color"]], df)
    assert "group" in out.columns


# -- stat_smooth -------------------------------------------------------------

def test_smooth_exact_line_recovers_fit_and_zero_band():
    x = np.linspace(0, 10, 25)
    df = pd.DataFrame({"x": x, "y": 3.0 * x - 2.0})
    out = stat_smooth(n=50).compute(df)
    assert list(out.columns[:5]) == ["x", "y", "ymin", "ymax", "se"]
    assert len(out) == 50
    slope, intercept = np.polyfit(out["x"], out["y"], 1)
    assert slope == pytest.approx(3.0)
    assert intercept == pytest.approx(-2.0)
    np.testing.assert_allclose(out["ymax"] - out["ymin"], 0.0, atol=1e-9)
    np.testing.assert_allclose(out["se"], 0.0, atol=1e-9)
    assert out["x"].min() == 0 and out["x"].max() == 10


def test_smooth_band_matches_textbook_lm():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 30)
    y = 1 + 2 * x + rng.normal(0, 0.3, 30)
    out = stat_smooth(n=5, level=0.9).compute(pd.DataFrame({"x": x, "y": y}))
    # textbook formula for simple linear regression
    b1, b0 = np.polyfit(x, y, 1)
    resid = y - (b0 + b1 * x)
    s = np.sqrt(resid @ resid / 28)
    x0 = out["x"].to_numpy()
    se = s * np.sqrt(1 / 30 + (x0 - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
    np.testing.assert_allclose(out["y"], b0 + b1 * x0)
    np.testing.assert_allclose(out["se"], se)
    from scipy import stats as st
    q = st.t.ppf(0.95, 28)
    np.testing.assert_allclose(out["ymax"] - out["y"], q * se)


def test_smooth_normal_fallback_without_scipy(monkeypatch):
    monkeypatch.setattr(smooth_mod, "_scipy_stats", None)
    assert smooth_mod.t_quantile(0.975, 5) == pytest.approx(1.959964, abs=1e-5)
    rng = np.random.default_rng(1)
    x = np.arange(20.0)
    df = pd.DataFrame({"x": x, "y": x + rng.normal(0, 1, 20)})
    out = stat_smooth().compute(df)
    np.testing.assert_allclose((out["ymax"] - out["y"]) / out["se"], 1.959964, rtol=1e-5)


def test_smooth_scipy_quantile():
    pytest.importorskip("scipy")
    assert smooth_mod.t_quantile(0.975, 5) == pytest.approx(2.570582, abs=1e-5)


def test_smooth_grouped_fits_per_group_and_carries_columns():
    x = np.tile(np.arange(10.0), 2)
    color = np.repeat(["a", "b"], 10)
    y = np.where(color == "a", 2 * x + 1, -x + 5)
    df = pd.DataFrame({"x": x, "y": y, "color": color, "PANEL": 1})
    out = stat_smooth(n=10, se=False).compute(df)
    assert set(out.columns) == {"x", "y", "color", "group", "PANEL"}
    assert sorted(out["group"].unique()) == [1, 2]
    for col, (m, b) in {"a": (2, 1), "b": (-1, 5)}.items():
        g = out[out["color"] == col]
        assert len(g) == 10
        assert (g["PANEL"] == 1).all()
        np.testing.assert_allclose(np.polyfit(g["x"], g["y"], 1), [m, b], atol=1e-9)


def test_smooth_polynomial_degree():
    x = np.linspace(-2, 2, 15)
    df = pd.DataFrame({"x": x, "y": x**2 - x + 4})
    out = stat_smooth(degree=2, n=7).compute(df)
    np.testing.assert_allclose(out["y"], out["x"] ** 2 - out["x"] + 4, atol=1e-9)
    np.testing.assert_allclose(out["se"], 0, atol=1e-9)


def test_smooth_fullrange_uses_scale_limits():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    out = stat_smooth(fullrange=True, n=11).compute(df, {"x": FakeScale((0, 10))})
    assert out["x"].min() == 0 and out["x"].max() == 10
    # missing / untrained scale falls back to the data range
    out = stat_smooth(fullrange=True).compute(df, {"x": FakeScale(None)})
    assert out["x"].min() == 1 and out["x"].max() == 3


def test_smooth_drops_degenerate_groups_and_na():
    df = pd.DataFrame({
        "x": [1.0, 1.0, 1.0, 2.0, np.nan],
        "y": [1.0, 2.0, 3.0, 4.0, 9.0],
        "color": ["a", "a", "b", "b", "b"],
    })
    out = stat_smooth(n=3).compute(df)
    assert out["color"].unique().tolist() == ["b"]
    assert out["se"].isna().all()  # 2 points: no residual dof


def test_smooth_requires_aes_and_known_method():
    with pytest.raises(ValueError, match="y"):
        stat_smooth().compute(pd.DataFrame({"x": [1, 2]}))
    with pytest.raises(NotImplementedError):
        stat_smooth(method="loess").compute(pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]}))


def test_compute_layer_ignores_unknown_params():
    df = add_group(pd.DataFrame({"x": [0.0, 1, 2], "y": [0.0, 1, 2]}))
    out = StatSmooth().compute_layer(df, None, n=4, se=False, alpha=0.5, color="red")
    assert len(out) == 4 and "se" not in out.columns


# -- stat_function -----------------------------------------------------------

def test_function_vectorised_with_xlim():
    out = stat_function(np.sin, n=11, xlim=(0, np.pi)).compute(pd.DataFrame())
    assert len(out) == 11
    np.testing.assert_allclose(out["y"], np.sin(out["x"]))
    assert (out["group"] == NO_GROUP).all()


def test_function_scalar_only_fun():
    def f(v):  # not array-safe
        return math.exp(v) if v > 0 else 0.0

    out = stat_function(f, n=5, xlim=(-1, 1)).compute(None)
    np.testing.assert_allclose(out["y"], [0, 0, 0, math.exp(0.5), math.e])


def test_function_range_from_scale_then_data():
    out = stat_function(lambda x: 2 * x, n=3).compute(
        pd.DataFrame({"x": [5.0, 7.0]}), {"x": FakeScale((0.0, 4.0))}
    )
    assert out["x"].tolist() == [0.0, 2.0, 4.0]
    out = stat_function(lambda x: 2 * x, n=3).compute(pd.DataFrame({"x": [5.0, 7.0]}))
    assert out["x"].tolist() == [5.0, 6.0, 7.0]
    with pytest.raises(ValueError, match="x range"):
        stat_function(np.sin).compute(pd.DataFrame())


def test_function_args_and_transformed_scales():
    scales = {"x": FakeScale((0.0, 2.0), Log10), "y": FakeScale(None, Log10)}
    out = stat_function(lambda x, k: k * x, n=3, args={"k": 10}).compute(None, scales)
    # x grid in log space 0,1,2 -> data 1,10,100 -> y 10,100,1000 -> log 1,2,3
    np.testing.assert_allclose(out["y"], [1, 2, 3])


def test_function_per_panel():
    df = pd.DataFrame({"x": [0.0, 1.0, 10.0, 20.0], "PANEL": [1, 1, 2, 2]})
    out = stat_function(lambda x: x, n=2).compute(df)
    assert out["PANEL"].tolist() == [1, 1, 2, 2]
    assert out["x"].tolist() == [0.0, 1.0, 10.0, 20.0]


def test_function_requires_callable():
    with pytest.raises(TypeError):
        stat_function(3)


# -- API / misc ----------------------------------------------------------------

def test_params_and_get_stat():
    s = stat_smooth(method="lm", se=False, n=10)
    assert s.params["se"] is False and s.params["n"] == 10
    assert get_stat(s) is s
    assert isinstance(get_stat("smooth"), StatSmooth)
    assert isinstance(get_stat("stat_identity"), Stat)
    with pytest.raises(ValueError):
        get_stat("nope")


def test_add_to_not_implemented():
    with pytest.raises(NotImplementedError, match="geom_"):
        stat_identity().add_to(object())


def test_stats_do_not_import_manim():
    import pathlib
    import subprocess
    import sys

    # Import ggmanim.stats without running ggmanim/__init__.py (which pulls in manim).
    code = (
        "import sys, types, pathlib, importlib;"
        "pkg = types.ModuleType('ggmanim'); pkg.__path__ = [str(pathlib.Path('ggmanim').resolve())];"
        "sys.modules['ggmanim'] = pkg;"
        "importlib.import_module('ggmanim.stats');"
        "assert 'manim' not in sys.modules, 'manim imported'"
    )
    root = pathlib.Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-c", code], check=True, cwd=root)
