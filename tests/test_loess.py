"""loess against R (fixtures from tests/fixtures/loess_r.R) and stat_smooth's
loess / auto / callable methods."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ggmanim import aes, geom_point, geom_smooth, ggplot
from ggmanim.stats import StatSmooth, stat_smooth
from ggmanim.stats import smooth as smooth_mod
from ggmanim.stats._loess import loess

FIXTURES = Path(__file__).parent / "fixtures"
DATA = pd.read_csv(FIXTURES / "loess_data.csv")
REF = pd.read_csv(FIXTURES / "loess_r.csv")
RTOL = 1e-3


class FakeScale:
    def __init__(self, limits):
        self._limits = limits

    def get_limits(self):
        return self._limits


# -- reference fixture --------------------------------------------------------


@pytest.mark.parametrize("degree", [1, 2])
def test_direct_surface_matches_r_exact(degree):
    # R: loess(y ~ x, span = .75, degree, control = loess.control("direct", "exact"))
    m = loess(DATA.x, DATA.y, span=0.75, degree=degree, surface="direct")
    fit, se = m.predict(REF.x, se=True)
    np.testing.assert_allclose(fit, REF[f"fit{degree}"], rtol=RTOL)
    np.testing.assert_allclose(se, REF[f"se{degree}"], rtol=RTOL)
    assert m.s == pytest.approx(REF[f"s{degree}"][0], rel=RTOL)
    assert m.df == pytest.approx(REF[f"df{degree}"][0], rel=RTOL)


def test_interpolated_surface_matches_r_default():
    # R's default (kd tree + cubic interpolation) is what ggplot2 draws
    m = loess(DATA.x, DATA.y)
    fit, se = m.predict(REF.x, se=True)
    np.testing.assert_allclose(fit, REF["fit2_interp"], rtol=RTOL)
    # R approximates the residual degrees of freedom; ours are exact
    np.testing.assert_allclose(se, REF["se2_interp"], rtol=1e-2)
    assert m.df == pytest.approx(REF["df2_interp"][0], rel=1e-2)


def test_interpolation_differs_from_direct_near_data_gaps():
    # sanity: the fixture distinguishes the two surfaces
    assert np.max(np.abs(REF["fit2"] - REF["fit2_interp"])) > 0.05


def test_loess_errors_and_extrapolation():
    with pytest.raises(ValueError, match="span too small"):
        loess([0.0, 1, 2], [0.0, 1, 2], span=0.5, degree=2)
    with pytest.raises(ValueError, match="degree"):
        loess(DATA.x, DATA.y, degree=3)
    with pytest.raises(ValueError, match="surface"):
        loess(DATA.x, DATA.y, surface="kd")
    # a straight line is reproduced exactly, also outside the data (extrapolation)
    x = np.linspace(0, 1, 30)
    m = loess(x, 2 * x + 1)
    np.testing.assert_allclose(m.predict([-0.5, 0.5, 1.5]), [0.0, 2.0, 4.0], atol=1e-9)


# -- stat_smooth(method="loess") ---------------------------------------------


def test_stat_smooth_loess_matches_r_band():
    out = stat_smooth(method="loess", n=80, level=0.95).compute(DATA)
    np.testing.assert_allclose(out["x"], REF.x, rtol=1e-9)
    np.testing.assert_allclose(out["y"], REF["fit2_interp"], rtol=RTOL)
    half = (out["ymax"] - out["ymin"]) / 2
    if smooth_mod._scipy_stats is not None:
        from scipy import stats

        want = REF["se2_interp"] * stats.t.ppf(0.975, REF["df2_interp"][0])
        np.testing.assert_allclose(half, want, rtol=1e-2)


def test_stat_smooth_loess_span_degree_and_method_args():
    a = stat_smooth(method="loess", span=0.5, degree=1, se=False).compute(DATA)
    b = stat_smooth(method="loess", se=False, method_args={"span": 0.5, "degree": 1}).compute(DATA)
    np.testing.assert_allclose(a["y"], b["y"])
    direct = stat_smooth(method="loess", se=False, n=80, method_args={"surface": "direct"}).compute(
        DATA
    )
    np.testing.assert_allclose(direct["y"], REF["fit2"], rtol=RTOL)


def test_stat_smooth_loess_too_few_points_drops_group_with_warning():
    # group b: 3 points, floor(3 * 0.75) = 2 neighbours <= degree 2
    df = pd.DataFrame({"x": [0.0, 1, 2, 3, 10, 11, 12], "y": [0.0, 1, 0, 1, 5, 6, 5]})
    df["g"] = ["a"] * 4 + ["b"] * 3
    df["group"] = [1] * 4 + [2] * 3
    with pytest.warns(UserWarning, match="loess failed"):
        out = stat_smooth(method="loess", n=5, span=0.75).compute(df)
    assert set(out["g"]) == {"a"}


def test_fullrange_loess_extrapolates_over_scale_limits():
    out = stat_smooth(method="loess", fullrange=True, n=13).compute(
        DATA, {"x": FakeScale((-2.0, 12.0))}
    )
    assert out["x"].min() == -2 and out["x"].max() == 12
    assert np.isfinite(out[["y", "ymin", "ymax"]].to_numpy()).all()


# -- method="auto" ------------------------------------------------------------


def test_auto_picks_loess_below_1000_with_message():
    with pytest.warns(UserWarning, match="method = 'loess' and formula = 'y ~ x'"):
        out = stat_smooth(method="auto", se=False, n=80).compute(DATA)
    np.testing.assert_allclose(out["y"], REF["fit2_interp"], rtol=RTOL)


def test_auto_picks_lm_from_1000_rows():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 1000)
    df = pd.DataFrame({"x": x, "y": 3 * x + rng.normal(0, 0.1, 1000)})
    with pytest.warns(UserWarning, match="method = 'lm'"):
        auto = stat_smooth(method="auto", se=False, n=5).compute(df)
    lm = stat_smooth(method="lm", se=False, n=5).compute(df)
    np.testing.assert_allclose(auto["y"], lm["y"])


def test_lm_default_degree_is_still_1():
    df = pd.DataFrame({"x": [0.0, 1, 2, 3], "y": [0.0, 1, 4, 9]})
    out = stat_smooth(se=False, n=4).compute(df)
    np.testing.assert_allclose(np.diff(out["y"], 2), 0, atol=1e-12)  # a straight line


# -- callable methods -----------------------------------------------------------


def test_callable_method_fit_only():
    calls = {}

    def model(x, y, power=1):
        calls["n"] = len(x)
        coef = np.polyfit(x, y, power)
        return lambda xg: np.polyval(coef, xg)

    df = pd.DataFrame({"x": [0.0, 1, 2, 3], "y": [1.0, 2, 5, 10]})
    out = stat_smooth(method=model, method_args={"power": 2}, n=4).compute(df)
    assert calls["n"] == 4
    np.testing.assert_allclose(out["y"], [1, 2, 5, 10], atol=1e-9)
    assert "ymin" not in out.columns  # no se from the model: no band


def test_callable_method_with_se_and_predict_object():
    class Model:
        def __init__(self, x, y):
            self.mean = float(np.mean(y))

        def predict(self, xg):
            return np.full(len(xg), self.mean), np.full(len(xg), 0.5)

    df = pd.DataFrame({"x": [0.0, 1, 2], "y": [1.0, 2, 3]})
    out = stat_smooth(method=Model, n=3, level=0.95).compute(df)
    np.testing.assert_allclose(out["y"], 2.0)
    np.testing.assert_allclose(out["ymax"] - out["y"], 0.5 * 1.959964, rtol=1e-6)
    np.testing.assert_allclose(out["se"], 0.5)


def test_callable_method_failure_warns_and_bad_length_raises():
    def broken(x, y):
        raise RuntimeError("boom")

    df = pd.DataFrame({"x": [0.0, 1, 2], "y": [1.0, 2, 3]})
    with pytest.warns(UserWarning, match="broken failed"):
        assert stat_smooth(method=broken).compute(df).empty
    with pytest.raises(ValueError, match="returned 2 values"):
        stat_smooth(method=lambda x, y: lambda g: g[:2], n=5).compute(df)


def test_unknown_method_still_rejected():
    with pytest.raises(NotImplementedError, match="gam"):
        StatSmooth(method="gam").compute(DATA)


# -- through the pipeline ---------------------------------------------------------


def test_geom_smooth_loess_and_callable_build():
    p = (
        ggplot(DATA, aes("x", "y"))
        .add(geom_point())
        .add(geom_smooth(method="loess", span=0.5), name="loess")
        .add(
            geom_smooth(method=lambda x, y: lambda g: np.full(len(g), y.mean()), se=False),
            name="mean",
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "unknown parameter" warnings
        m = p.build()
    assert len(m.layers["loess"].marks) == 1
    assert len(m.layers["mean"].marks) == 1
