import numpy as np
import pandas as pd

import ggmanim as gg
from ggmanim.geoms.vline import GeomHline, GeomVline


def _plot(theta, line):
    df = pd.DataFrame({"x": [0.0, 1, 2, 3, 4], "y": [1.0, 2, 3, 2, 1]})
    return gg.ggplot(df, gg.aes("x", "y")).add(gg.geom_point(), line, gg.coord_radar(theta=theta))


def _extent(plot):
    m = plot.build()
    paths = [s for s in m.get_family() if type(s).__name__ == "VMobject" and len(s.points) > 4]
    pts = max(paths, key=lambda s: len(s.points)).points
    return np.ptp(pts[:, 0]), np.ptp(pts[:, 1])


def test_hline_is_polygon_under_radar():
    w, h = _extent(_plot("x", gg.geom_hline(yintercept=2)))
    assert w > 0.5 and h > 0.5


def test_vline_is_polygon_when_theta_is_y():
    w, h = _extent(_plot("y", gg.geom_vline(xintercept=2)))
    assert w > 0.5 and h > 0.5


def test_spokes_only_along_theta():
    coord = gg.coord_radar()
    pp = coord.setup_panel_params(None, None)
    pp = pp.__class__(**{**pp.__dict__, "x_breaks": (0.0, 1.0, 2.0)})
    assert GeomVline()._radar_spokes(coord, pp) is None
    assert list(GeomHline()._radar_spokes(coord, pp)) == [0.0, 1.0, 2.0]


def test_ring_closed():
    x, y = GeomHline()._ring(np.array([0.0, 1, 2]), 5.0)
    assert x[0] == x[-1] and list(y) == [5.0] * 4
