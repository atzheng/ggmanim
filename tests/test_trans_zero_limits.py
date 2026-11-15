"""coord_trans on a log axis whose limits include 0 (ggm-un7)."""

import numpy as np
import pandas as pd
import pytest

from ggmanim import aes, coord_trans, ggplot
from ggmanim.geoms.base import munch_rect_corners


def _col(**kw):
    df = pd.DataFrame({"x": list("abc"), "y": [10.0, 100.0, 1000.0]})
    return ggplot(df, aes("x", "y")).geom_col().add(coord_trans(y="log10", **kw)).build()


def test_log_axis_keeps_finite_data_range_when_limits_include_zero():
    lo, hi = _col().panel_params.trans_y_range
    assert lo < 1.0 and hi > 3.0  # covers log10(10) .. log10(1000), expanded
    assert hi - lo > 2.0  # not collapsed to +-0.5 around one end


def test_munched_rect_from_zero_stops_at_panel_edge_clip_off():
    p = _col(clip="off")
    edge = p.panel.transform(1, p.panel_params.y_range[0])[0, 1]
    ring = munch_rect_corners(p.panel, [[0.6, 1.4, 1.4, 0.6]], [[0.0, 0.0, 10.0, 10.0]])
    assert np.isfinite(ring).all()
    assert ring[0, :, 1].min() == pytest.approx(edge, abs=1e-6)
    for m in p.layers[0].marks:
        pts = m.get_points()
        assert np.isfinite(pts).all()
        assert pts[:, 1].min() == pytest.approx(edge, abs=1e-6)
