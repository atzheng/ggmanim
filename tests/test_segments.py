"""Per-segment drawing of path geoms whose colour/linewidth/alpha vary in a group."""

import numpy as np
import pandas as pd
import pytest
from manim import BLUE, GREEN, RED, VGroup, VMobject

from ggmanim import aes, coord_cartesian, geom_line, ggplot, scale_color_viridis_c
from ggmanim.coords import CoordCartesian
from ggmanim.geoms import GeomLine, GeomPath
from ggmanim.geoms._segments import dash_polyline_phase, varies_within
from ggmanim.scales import scale_x_continuous, scale_y_continuous
from ggmanim.theme import theme_default


def _panel(x, y, **coord):
    sx, sy = scale_x_continuous(), scale_y_continuous()
    sx.train(np.asarray(x, dtype=float))
    sy.train(np.asarray(y, dtype=float))
    return CoordCartesian(**coord).build(sx, sy, theme_default())


def _hex(m: VMobject) -> str:
    return m.get_stroke_color().to_hex().upper()


def _verts(m: VMobject) -> np.ndarray:
    return np.vstack([m.get_start_anchors(), m.get_end_anchors()[-1:]])


H = {c: c.to_hex().upper() for c in (RED, GREEN, BLUE)}


def test_varies_within():
    df = pd.DataFrame({"color": [RED, RED], "linewidth": [3, 3], "alpha": [1, 1]})
    assert not varies_within(df)
    assert varies_within(df.assign(color=[RED, BLUE]))
    assert varies_within(df.assign(alpha=[1.0, 0.5]))
    assert not varies_within(df.iloc[:1].assign(color=[BLUE]))
    with pytest.raises(ValueError, match="linetype must be constant"):
        varies_within(df.assign(linetype=["solid", "dashed"], group=1))


def test_varying_colour_one_segment_per_vertex_pair_styled_by_start():
    data = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 0.0, 1.0], "color": [RED, GREEN, BLUE, BLUE]}
    ).assign(group=1)
    panel = _panel(data.x, data.y)
    out = GeomPath().draw_panel(data, panel.panel_params, panel)
    (segs,) = out.marks
    assert isinstance(segs, VGroup)
    # RED: 0->1, GREEN: 1->2, BLUE: 2->3 (last vertex's colour unused)
    assert [_hex(s) for s in segs] == [H[RED], H[GREEN], H[BLUE]]
    pts = panel.transform(data.x, data.y)
    for k, s in enumerate(segs):
        v = _verts(s)
        np.testing.assert_allclose(v[0], pts[k], atol=1e-9)
        np.testing.assert_allclose(v[1], pts[k + 1], atol=1e-9)
        # opaque runs overlap a little into the next segment (anti-aliasing seam)
        assert len(v) == (3 if k < 2 else 2)
        if k < 2:
            d = v[2] - pts[k + 1]
            assert 0 < np.linalg.norm(d) <= 0.03 + 1e-9
            assert np.linalg.norm(np.cross(d, pts[k + 2] - pts[k + 1])) < 1e-9


def test_same_style_runs_merge_and_linewidth_alpha_vary():
    data = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [0.0, 1.0, 0.0, 1.0, 0.0],
            "color": [RED] * 5,
            "linewidth": [2.0, 2.0, 2.0, 6.0, 6.0],
            "alpha": [0.5] * 5,
        }
    ).assign(group=1)
    panel = _panel(data.x, data.y)
    (segs,) = GeomLine().draw_panel(data, panel.panel_params, panel).marks
    assert [s.get_stroke_width() for s in segs] == [2.0, 6.0]
    assert [len(_verts(s)) for s in segs] == [4, 2]  # translucent: no overlap
    assert all(s.get_stroke_opacity() == pytest.approx(0.5) for s in segs)


def test_constant_colour_group_unchanged_single_path():
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 0.0], "color": [RED] * 3})
    panel = _panel(data.x, data.y)
    (path,) = GeomLine().draw_panel(data.assign(group=1), panel.panel_params, panel).marks
    assert type(path) is VMobject and len(_verts(path)) == 3 and _hex(path) == H[RED]


def test_varying_colour_clipped():
    data = pd.DataFrame(
        {"x": [0.0, 2.0, 4.0], "y": [0.0, 2.0, 0.0], "color": [RED, BLUE, GREEN]}
    ).assign(group=1)
    panel = _panel([0.0, 4.0], [0.0, 1.0], ylim=(0.0, 1.0), expand=False)
    (segs,) = GeomLine().draw_panel(data, panel.panel_params, panel).marks
    assert [_hex(s) for s in segs] == [H[RED], H[BLUE]]
    exp = panel.transform([0.0, 1.0, 3.0, 4.0], [0.0, 1.0, 1.0, 0.0])
    for s, (a, b) in zip(segs, [(exp[0], exp[1]), (exp[2], exp[3])]):
        v = _verts(s)
        assert len(v) == 2  # cut at the border: no overlap, no join
        np.testing.assert_allclose(v, [a, b], atol=1e-9)


def test_dash_polyline_phase_continues_pattern():
    pts = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    d = dash_polyline_phase(pts, (1.0, 1.0), 0.0)
    assert len(d) == 5 and d[0][0, 0] == 0 and d[0][-1, 0] == 1
    d = dash_polyline_phase(pts, (1.0, 1.0), 1.5)  # starts in a gap
    assert d[0][0, 0] == pytest.approx(0.5) and d[0][-1, 0] == pytest.approx(1.5)
    d = dash_polyline_phase(pts, (1.0, 1.0), 0.25)  # starts inside a dash
    assert d[0][0, 0] == 0 and d[0][-1, 0] == pytest.approx(0.75)


def test_dashed_varying_colour_pattern_continues_across_segments():
    data = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0], "color": [RED, BLUE, BLUE]}
    ).assign(group=1)
    panel = _panel([0.0, 2.0], [-1.0, 1.0], expand=False)
    out = GeomLine().draw_panel(data, panel.panel_params, panel, linetype="dashed", linewidth=3)
    (dashes,) = out.marks
    assert dashes.linetype == "dashed"
    xs = sorted((_verts(m)[0, 0], _verts(m)[-1, 0]) for m in dashes if _hex(m) == H[BLUE])
    # the dashes of the BLUE half lie on the whole-path grid that the RED half starts
    unit = 0.01 * 3 * 4  # "dashed" = (4, 4) line-width units
    x0 = panel.transform([0.0], [0.0])[0, 0]
    phases = [((a - x0) / unit) % 2 for a, _ in xs]
    mid = panel.transform([1.0], [0.0])[0, 0]
    assert len(xs) >= 3
    assert all(min(p, 2 - p) < 1e-6 or a == pytest.approx(mid) for p, (a, _) in zip(phases, xs))
    assert {_hex(m) for m in dashes} == {H[RED], H[BLUE]}


def test_viridis_line_build_end_to_end():
    t = np.linspace(0, 1, 11)
    p = ggplot(pd.DataFrame({"x": t, "y": t**2, "t": t}), aes("x", "y", color="t")).add(
        geom_line(), scale_color_viridis_c(), coord_cartesian()
    )
    plot = p.build()
    (layer,) = plot.layers.values() if hasattr(plot.layers, "values") else plot.layers
    (segs,) = layer.marks
    assert len(segs) == 10
    assert len({_hex(s) for s in segs}) == 10
