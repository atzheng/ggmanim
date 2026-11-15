"""hexbin and summary_hex share one lattice helper; ordinal shape/linetype warn."""

import warnings

import numpy as np
import pandas as pd
import pytest

from ggmanim.scales import default_scale_for
from ggmanim.stats.hex import _hex_cells, hexbin
from ggmanim.stats.summary_hex import hex_cells


def _legacy_cells(x, y, binwidth, xlim, ylim):
    """The pre-dedupe lattice assignment, inlined as the reference."""
    s3 = np.sqrt(3.0)
    wx, wy = binwidth
    xlo = np.floor(xlim[0] / wx) * wx - 1e-6
    ylo = np.floor(ylim[0] / wy) * wy - 1e-6
    sx, sy = (x - xlo) / wx, (y - ylo) / (s3 * wy)
    i1, j1 = np.floor(sx + 0.5), np.floor(sy + 0.5)
    i2, j2 = np.floor(sx) + 0.5, np.floor(sy) + 0.5
    first = (sx - i1) ** 2 + 3 * (sy - j1) ** 2 <= (sx - i2) ** 2 + 3 * (sy - j2) ** 2
    return np.where(first, i1, i2), np.where(first, j1, j2)


def test_shared_helper_bit_identical():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=500), rng.normal(size=500) * 2
    bw = (0.37, 0.61)
    xl, yl = (x.min(), x.max()), (y.min(), y.max())
    ref = _legacy_cells(x, y, bw, xl, yl)
    for got in (_hex_cells(x, y, bw, xl, yl), hex_cells(x, y, bw, xl, yl)):
        assert np.array_equal(got[0], ref[0]) and np.array_equal(got[1], ref[1])
    out = hexbin(x, y, bw)
    s3 = np.sqrt(3.0)
    xlo = np.floor(xl[0] / bw[0]) * bw[0] - 1e-6
    ylo = np.floor(yl[0] / bw[1]) * bw[1] - 1e-6
    cells = pd.DataFrame({"cj": ref[1], "ci": ref[0], "count": 1.0})
    exp = cells.groupby(["cj", "ci"]).sum().reset_index()
    assert np.array_equal(out["x"], xlo + exp["ci"] * bw[0])
    assert np.array_equal(out["y"], ylo + exp["cj"] * s3 * bw[1])
    assert np.array_equal(out["count"], exp["count"])


def test_bad_binwidth_raises():
    with pytest.raises(ValueError):
        hexbin([0.0, 1.0], [0.0, 1.0], (0.0, 1.0))


@pytest.mark.parametrize("aes", ["shape", "linetype"])
def test_ordinal_shape_linetype_warn(aes):
    vals = pd.Series(pd.Categorical(["a", "b"], ordered=True))
    with pytest.warns(UserWarning, match="ordinal variable is not advised"):
        default_scale_for(aes, vals)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        default_scale_for(aes, pd.Series(["a", "b"]))
