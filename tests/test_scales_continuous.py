import numpy as np
import pandas as pd
import pytest

from ggmanim.scales import (
    Scale,
    ScaleContinuous,
    get_transform,
    scale_x_continuous,
    scale_x_log10,
    scale_x_reverse,
    scale_x_sqrt,
    scale_y_continuous,
    scale_y_log10,
    scale_y_reverse,
    scale_y_sqrt,
    xlim,
    ylim,
)


# ----------------------------------------------------------------- transforms
@pytest.mark.parametrize("name", ["identity", "log10", "log2", "log", "sqrt", "reverse"])
def test_transform_round_trip(name):
    t = get_transform(name)
    x = np.array([0.01, 0.5, 1.0, 3.0, 2500.0, 1e6])
    np.testing.assert_allclose(t.inverse(t.transform(x)), x, rtol=1e-12)


def test_transform_domain_gives_nan():
    with np.errstate(all="raise"):  # no floating-point warnings leak out
        assert np.isnan(get_transform("log10").transform([0.0, -1.0])).all()
        assert np.isnan(get_transform("sqrt").transform([-1.0])).all()


def test_unknown_transform():
    with pytest.raises(ValueError):
        get_transform("bogus")


# ----------------------------------------------------------------- base API
def test_scale_is_abstract():
    with pytest.raises(TypeError):
        Scale("x")


def test_constructors_aesthetics():
    assert scale_x_continuous().aesthetics[0] == "x"
    assert "xintercept" in scale_x_continuous().aesthetics
    assert scale_y_log10().aesthetic == "y"
    assert scale_y_log10().has_aesthetic("yintercept")
    for f in (scale_x_continuous, scale_x_log10, scale_x_sqrt, scale_x_reverse):
        assert f().aesthetic == "x" and isinstance(f(), ScaleContinuous)
    for f in (scale_y_continuous, scale_y_log10, scale_y_sqrt, scale_y_reverse):
        assert f().aesthetic == "y"
    assert scale_x_continuous(name="Dose").name == "Dose"
    assert not scale_x_continuous().is_discrete


def test_untrained_scale_is_empty():
    s = scale_x_continuous()
    assert s.is_empty()
    assert s.get_limits() is None
    assert s.get_expanded_limits() is None
    assert len(s.get_breaks()) == 0 and s.get_labels() == []


def test_train_accumulates_and_ignores_nan():
    s = scale_x_continuous()
    s.train([1, 5, np.nan])
    s.train(pd.Series([-2.0, 3.0]))
    s.train([np.inf])
    s.train([])
    assert s.get_limits() == (-2.0, 5.0)


def test_train_rejects_strings():
    with pytest.raises(TypeError):
        scale_x_continuous().train(["a", "b"])


def test_clone_and_reset():
    s = scale_x_continuous(limits=(None, 10), name="x")
    s.train([0, 5])
    c = s.clone()
    assert c.range is None and c.name == "x" and c.limits == (None, 10)
    assert s.range == (0.0, 5.0)  # original untouched
    c.train([1, 2])
    assert s.range == (0.0, 5.0)
    s.reset()
    assert s.range is None and s.limits == (None, 10)


def test_add_to_uses_plot_with_scale():
    class FakePlot:
        def _with_scale(self, scale):
            return ("added", scale)

    s = scale_x_log10()
    assert s.add_to(FakePlot()) == ("added", s)


# ----------------------------------------------------------------- continuous
def test_continuous_limits_breaks_labels():
    s = scale_x_continuous()
    s.train([0.0, 1.0])
    assert s.get_limits() == (0.0, 1.0)
    assert s.get_expanded_limits() == pytest.approx((-0.05, 1.05))
    assert s.dimension() == s.get_expanded_limits()
    np.testing.assert_allclose(s.get_breaks(), [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert s.get_labels() == ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]


def test_breaks_are_censored_to_limits():
    s = scale_y_continuous()
    s.train([-17, 42])
    np.testing.assert_allclose(s.get_breaks(), [-10, 0, 10, 20, 30, 40])
    # the coord may ask for breaks over the expanded range instead
    lo, hi = s.get_expanded_limits()
    b = s.get_breaks(s.get_expanded_limits())
    assert b.min() >= lo and b.max() <= hi
    assert len(s.get_labels(s.get_expanded_limits())) == len(b)


def test_log10_scale():
    s = scale_x_log10()
    s.train([3, 2500])
    lo, hi = s.get_limits()
    assert (lo, hi) == pytest.approx((np.log10(3), np.log10(2500)))
    elo, ehi = s.get_expanded_limits()
    w = hi - lo
    assert (elo, ehi) == pytest.approx((lo - 0.05 * w, hi + 0.05 * w))
    np.testing.assert_allclose(s.get_breaks(), [1, 2, 3])
    assert s.get_labels() == ["10", "100", "1000"]
    np.testing.assert_allclose(s.map([10, 100]), [1, 2])


def test_log10_scale_non_positive_values_dropped():
    s = scale_x_log10()
    s.train([0, -1, 10, 1000])
    assert s.get_limits() == pytest.approx((1, 3))
    assert np.isnan(s.map([0])[0])


def test_log10_narrow_range_uses_sub_breaks():
    s = scale_y_log10()
    s.train([3, 8])
    assert s.get_labels() == ["3", "4", "5", "6", "7", "8"]


def test_sqrt_scale():
    s = scale_x_sqrt()
    s.train([0, 100])
    assert s.get_limits() == (0.0, 10.0)
    assert s.get_labels() == ["0", "20", "40", "60", "80", "100"]
    np.testing.assert_allclose(s.get_breaks(), np.sqrt([0, 20, 40, 60, 80, 100]))


def test_reverse_scale():
    s = scale_y_reverse()
    s.train([0, 10])
    assert s.get_limits() == (-10.0, 0.0)
    np.testing.assert_allclose(s.map([0, 5, 10]), [0, -5, -10])
    assert s.get_labels() == ["0", "2", "4", "6", "8", "10"]
    np.testing.assert_allclose(s.get_breaks(), [0, -2, -4, -6, -8, -10])


def test_user_limits_censor():
    s = scale_x_continuous(limits=(0, 5))
    s.train([1, 10])
    assert s.get_limits() == (0.0, 5.0)
    out = s.map([-1, 0, 2.5, 5, 6, np.nan])
    np.testing.assert_array_equal(np.isnan(out), [True, False, False, False, True, True])


def test_oob_squish_and_keep():
    s = scale_x_continuous(limits=(0, 5), oob="squish")
    np.testing.assert_allclose(s.map([-1, 6]), [0, 5])
    s = scale_x_continuous(limits=(0, 5), oob="keep")
    np.testing.assert_allclose(s.map([-1, 6]), [-1, 6])
    s = scale_x_continuous(limits=(0, 5), oob=lambda x, r: x * 0)
    np.testing.assert_allclose(s.map([-1, 6]), [0, 0])
    with pytest.raises(ValueError):
        scale_x_continuous(oob="nope")


def test_partial_limits():
    s = scale_x_continuous(limits=(None, 5))
    s.train([1, 10])
    assert s.get_limits() == (1.0, 5.0)
    s = scale_x_reverse(limits=(None, 5))
    s.train([1, 10])
    assert s.get_limits() == (-5.0, -1.0)


def test_log_limits_in_data_space():
    s = scale_x_log10(limits=(1, 1000))
    s.train([5, 50])
    assert s.get_limits() == pytest.approx((0, 3))
    assert s.get_labels() == ["1", "10", "100", "1000"]
    assert np.isnan(s.map([5000])[0])
    with pytest.raises(ValueError):
        scale_x_log10(limits=(0, 10)).get_limits()


def test_user_breaks_and_labels():
    s = scale_x_continuous(breaks=[0, 5, 10, 50], labels=["a", "b", "c", "d"])
    s.train([0, 12])
    np.testing.assert_allclose(s.get_breaks(), [0, 5, 10])
    assert s.get_labels() == ["a", "b", "c"]
    s = scale_x_log10(breaks=[1, 10, 100])
    s.train([1, 100])
    np.testing.assert_allclose(s.get_breaks(), [0, 1, 2])
    assert s.get_labels() == ["1", "10", "100"]


def test_callable_breaks_and_labels():
    s = scale_x_continuous(
        breaks=lambda lim: np.arange(lim[0], lim[1] + 1, 3),
        labels=lambda b: [f"{v:g}%" for v in b],
    )
    s.train([0, 9])
    np.testing.assert_allclose(s.get_breaks(), [0, 3, 6, 9])
    assert s.get_labels() == ["0%", "3%", "6%", "9%"]


def test_no_breaks():
    s = scale_x_continuous(breaks=[])
    s.train([0, 1])
    assert len(s.get_breaks()) == 0 and s.get_labels() == []


def test_labels_length_mismatch():
    with pytest.raises(ValueError):
        scale_x_continuous(breaks=[1, 2], labels=["a"])


def test_expand_options():
    s = scale_x_continuous(expand=(0, 0))
    s.train([0, 10])
    assert s.get_expanded_limits() == (0, 10)
    s = scale_x_continuous(expand=(0, 1))
    s.train([0, 10])
    assert s.get_expanded_limits() == (-1, 11)
    assert s.get_expanded_limits(expand=False) == (0, 10)


def test_single_value_range():
    s = scale_x_continuous()
    s.train([3, 3])
    assert s.get_expanded_limits() == (2.5, 3.5)
    assert len(s.get_breaks()) >= 1


# ----------------------------------------------------------------- xlim / ylim
def test_xlim_ylim():
    s = xlim(0, 10)
    assert s.aesthetic == "x" and s.limits == (0, 10)
    s = ylim((0, 10))
    assert s.aesthetic == "y" and s.limits == (0, 10)
    s.train([5, 20])
    assert np.isnan(s.map([20])[0])


def test_xlim_reversed_gives_reverse_scale():
    s = xlim(10, 0)
    assert s.trans.name == "reverse"
    s.train([2, 3])
    assert s.get_limits() == (-10.0, 0.0)


def test_xlim_rejects_three_values_and_strings():
    with pytest.raises(ValueError, match="breaks"):
        xlim(0, 10, 2)
    with pytest.raises(ValueError):
        scale_x_continuous(limits=(0, 10, 2))
    with pytest.raises(TypeError):
        xlim("a", "b")


def test_map_does_not_require_training():
    s = scale_x_log10()
    np.testing.assert_allclose(s.map([10, 100]), [1, 2])
