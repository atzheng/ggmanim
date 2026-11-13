import numpy as np
import pytest

from ggmanim.scales.breaks import (
    expand_range,
    expansion,
    format_numbers,
    log_breaks,
    pretty,
)


# Expected values are R's pretty() output.
@pytest.mark.parametrize(
    "lo, hi, expected",
    [
        (0, 1, [0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        (0.003, 0.07, [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]),
        (-17, 42, [-20, -10, 0, 10, 20, 30, 40, 50]),
        (0, 100, [0, 20, 40, 60, 80, 100]),
        (1, 10, [0, 2, 4, 6, 8, 10]),
        (3, 2500, [0, 500, 1000, 1500, 2000, 2500]),
        (0.5, 1.7, [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]),
        (-1, 1, [-1.0, -0.5, 0.0, 0.5, 1.0]),
        (1e5, 1e6, [0, 2e5, 4e5, 6e5, 8e5, 1e6]),
        (1234, 1267, [1230, 1235, 1240, 1245, 1250, 1255, 1260, 1265, 1270]),
    ],
)
def test_pretty_matches_r(lo, hi, expected):
    np.testing.assert_allclose(pretty(lo, hi), expected, atol=1e-12)


def test_pretty_no_float_noise():
    assert pretty(0, 1).tolist() == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def test_pretty_covers_range_and_is_nice():
    rng = np.random.default_rng(0)
    for _ in range(200):
        lo, hi = sorted(rng.uniform(-1e3, 1e3, 2))
        b = pretty(lo, hi)
        assert b[0] <= lo + 1e-9 and b[-1] >= hi - 1e-9
        step = np.diff(b)
        assert np.allclose(step, step[0])
        mant = step[0] / 10 ** np.floor(np.log10(step[0]))
        assert any(np.isclose(mant, m) for m in (1, 2, 5, 10))


def test_pretty_degenerate():
    b = pretty(5, 5)
    assert b[0] <= 5 <= b[-1] and len(b) >= 2
    b = pretty(0, 0)
    assert b[0] <= 0 <= b[-1] and len(b) >= 2


def test_pretty_reversed_and_nonfinite():
    np.testing.assert_allclose(pretty(1, 0), pretty(0, 1))
    with pytest.raises(ValueError):
        pretty(0, np.inf)


def test_log_breaks_powers_of_ten():
    np.testing.assert_allclose(log_breaks(3, 2500), [1, 10, 100, 1000, 10000])


def test_log_breaks_sub_breaks_for_narrow_range():
    b = log_breaks(3, 30)
    inside = b[(b >= 3) & (b <= 30)]
    np.testing.assert_allclose(inside, [5, 10, 20])  # 1,5 too few -> 1,2,5 steps


def test_log_breaks_falls_back_to_pretty():
    np.testing.assert_allclose(log_breaks(20, 30), pretty(20, 30))


def test_log_breaks_rejects_nonpositive():
    with pytest.raises(ValueError):
        log_breaks(0, 10)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([0, 0.2, 0.4, 0.6000000000000001, 0.8, 1], ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]),
        ([10, 100, 1000], ["10", "100", "1000"]),
        ([0, 0.25, 0.5], ["0.00", "0.25", "0.50"]),
        ([-20, -10, -0.0, 10], ["-20", "-10", "0", "10"]),
        ([1e5, 2e5], ["1e+05", "2e+05"]),
        ([0, 2.5e5, 5e5], ["0.0e+00", "2.5e+05", "5.0e+05"]),
        ([1e-5, 2e-5], ["1e-05", "2e-05"]),
        ([99999], ["99999"]),
        ([np.nan, 1], ["", "1"]),
        # at most 7 significant digits (R's default), decimals still shared
        ([0.9999999999, 1.5, 2.0000000001], ["1.0", "1.5", "2.0"]),
        ([0.9999999999, 2.0000000001], ["1", "2"]),
        ([-1.23456789, 3.3], ["-1.234568", "3.300000"]),
        ([12345.6789123], ["12345.68"]),
        ([0.000123456789], ["0.0001234568"]),
        ([1.23456789e6, 2e6], ["1.234568e+06", "2.000000e+06"]),
        ([9999999.99], ["1e+07"]),
    ],
)
def test_format_numbers(values, expected):
    assert format_numbers(values) == expected


def test_expansion_and_expand_range():
    assert expansion(mult=0.05) == (0.05, 0.0, 0.05, 0.0)
    assert expansion(mult=(0, 0.1), add=1) == (0.0, 1.0, 0.1, 1.0)
    assert expand_range((0, 10)) == pytest.approx((-0.5, 10.5))
    assert expand_range((0, 10), (0, 1)) == pytest.approx((-1, 11))
    assert expand_range((0, 10), expansion(mult=(0, 0.1))) == pytest.approx((0, 11))
    assert expand_range((0, 10), False) == (0, 10)
    assert expand_range((3, 3)) == (2.5, 3.5)
    with pytest.raises(ValueError):
        expand_range((0, 1), (1, 2, 3))
