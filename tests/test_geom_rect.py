"""geom_rect, geom_tile, geom_raster: structure, clipping, flip, speed, raster == tile."""

import time

import numpy as np
import pandas as pd
import pytest
from manim import Camera, VGroup, VMobject

from ggmanim import (
    aes,
    coord_cartesian,
    coord_flip,
    geom_raster,
    geom_rect,
    geom_tile,
    ggplot,
    scale_fill_manual,
    scale_x_discrete,
    theme,
)
from ggmanim.geoms import KEY_SIZE, GeomRaster, GeomRect, GeomTile
from ggmanim.geoms._rect import ring_points, rings_mobject
from ggmanim.registry import COMPONENTS


def _corners(m: VMobject) -> np.ndarray:
    return np.asarray(m.get_start_anchors())


def _close(a, b, tol=1e-6):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol)


RECTS = pd.DataFrame({"x0": [0, 2], "x1": [1, 4], "y0": [0, 1], "y1": [2, 3]})


@pytest.mark.parametrize("name", ["geom_rect", "geom_tile", "geom_raster"])
def test_registered(name):
    assert name in COMPONENTS


def test_rect_vertices_and_marks():
    g = ggplot(RECTS, aes(xmin="x0", xmax="x1", ymin="y0", ymax="y1")).add(geom_rect()).build()
    mob = g.layers[0]
    assert len(mob.marks) == 2 and mob.keys == [0, 1]
    for m, row in zip(mob.marks, RECTS.itertuples()):
        want = g.panel.transform([row.x0, row.x1, row.x1, row.x0], [row.y0, row.y0, row.y1, row.y1])
        assert _close(_corners(m), want)
        assert m.get_fill_opacity() == pytest.approx(1.0)
        assert m.get_stroke_width() == 0  # no outline by default


def test_rect_alpha_fill_only_and_outline():
    g = (
        ggplot(RECTS, aes(xmin="x0", xmax="x1", ymin="y0", ymax="y1"))
        .add(geom_rect(fill="#FF0000", color="white", alpha=0.5, linewidth=2))
        .build()
    )
    m = g.layers[0].marks[0]
    assert m.get_fill_opacity() == pytest.approx(0.5)
    assert m.get_stroke_opacity() == pytest.approx(1.0)
    assert m.get_stroke_width() == pytest.approx(2)
    assert m.get_fill_color().to_hex().upper() == "#FF0000"


def test_rect_dashed_outline_is_group():
    g = (
        ggplot(RECTS, aes(xmin="x0", xmax="x1", ymin="y0", ymax="y1"))
        .add(geom_rect(color="white", linetype="dashed"))
        .build()
    )
    m = g.layers[0].marks[0]
    assert isinstance(m, VGroup) and len(m) == 2
    assert m[0].get_stroke_width() == 0 and len(m[1]) > 4  # fill + dashes


def test_rect_clipped_to_panel():
    g = (
        ggplot(RECTS, aes(xmin="x0", xmax="x1", ymin="y0", ymax="y1"))
        .add(geom_rect(), coord_cartesian(xlim=(0.5, 3), ylim=(0, 2.5), expand=False))
        .build()
    )
    mob = g.layers[0]
    assert len(mob.marks) == 2
    want = g.panel.transform([2, 3, 3, 2], [1, 1, 2.5, 2.5])
    assert _close(_corners(mob.marks[1]), want)
    # entirely outside -> dropped (and its key with it)
    g = (
        ggplot(RECTS, aes(xmin="x0", xmax="x1", ymin="y0", ymax="y1"))
        .add(geom_rect(), coord_cartesian(xlim=(1.5, 4), expand=False))
        .build()
    )
    assert g.layers[0].keys == [1]


def test_rect_coord_flip():
    g = (
        ggplot(RECTS, aes(xmin="x0", xmax="x1", ymin="y0", ymax="y1"))
        .add(geom_rect(), coord_flip())
        .build()
    )
    m = g.layers[0].marks[1]
    # x (2 -> 4) runs vertically under coord_flip
    c = _corners(m)
    assert c[1][0] == pytest.approx(c[0][0]) and c[1][1] > c[0][1]
    assert _close(c, g.panel.transform([2, 4, 4, 2], [1, 1, 3, 3]))


def test_draw_key_filled_square():
    for geom in (GeomRect(), GeomTile(), GeomRaster()):
        key = geom.draw_key({"fill": "#00FF00"})
        assert key.width == pytest.approx(KEY_SIZE)
        assert key.get_fill_color().to_hex().upper() == "#00FF00"


# -- tile ----------------------------------------------------------------------------


def test_tile_setup_data_resolution_and_params():
    df = pd.DataFrame({"x": [0.0, 2.0, 4.0], "y": [1.0, 1.5, 3.0]})
    out = GeomTile().setup_data(df, {})
    assert list(out["xmin"]) == [-1, 1, 3] and list(out["xmax"]) == [1, 3, 5]
    assert list(out["ymin"]) == [0.75, 1.25, 2.75]
    out = GeomTile().setup_data(df, {"width": 1.0, "height": 2.0})
    assert list(out["xmax"] - out["xmin"]) == [1, 1, 1]
    assert list(out["ymax"] - out["ymin"]) == [2, 2, 2]
    out = GeomTile().setup_data(df.assign(width=[1.0, 2.0, 3.0]), {})
    assert list(out["xmax"] - out["xmin"]) == [1, 2, 3]
    out = GeomTile().setup_data(df, {"_x_discrete": True})
    assert list(out["xmax"] - out["xmin"]) == [1, 1, 1]


def test_tile_discrete_axes_build():
    df = pd.DataFrame({"x": ["a", "b", "a", "b"], "y": ["p", "p", "q", "q"], "v": [1, 2, 3, 4]})
    g = ggplot(df, aes("x", "y", fill="v")).add(geom_tile()).build()
    data = g.layer_data[0]
    assert list(data["xmax"] - data["xmin"]) == [1, 1, 1, 1]
    assert list(data["ymax"] - data["ymin"]) == [1, 1, 1, 1]
    assert len(g.layers[0].marks) == 4
    # tiles abut: neighbours share an edge in the scene
    m0, m1 = g.layers[0].marks[:2]
    assert m0.get_right()[0] == pytest.approx(m1.get_left()[0], abs=1e-3)


def test_tile_seams_sealed_only_when_opaque():
    df = pd.DataFrame({"x": [0, 1], "y": [0, 0], "v": [1.0, 2.0]})
    m = ggplot(df, aes("x", "y", fill="v")).add(geom_tile()).build().layers[0].marks[0]
    assert m.get_stroke_width() > 0
    assert m.get_stroke_color().to_hex() == m.get_fill_color().to_hex()
    m = ggplot(df, aes("x", "y", fill="v")).add(geom_tile(alpha=0.5)).build().layers[0].marks[0]
    assert m.get_stroke_width() == 0


def test_tile_heatmap_50x50_builds_fast():
    n = 50
    xx, yy = np.meshgrid(np.arange(n), np.arange(n))
    df = pd.DataFrame({"x": xx.ravel(), "y": yy.ravel(), "z": np.sin(xx.ravel() * yy.ravel())})
    p = ggplot(df, aes("x", "y", fill="z")).add(geom_tile())
    t = time.perf_counter()
    g = p.build()
    assert time.perf_counter() - t < 2.0
    assert len(g.layers[0].marks) == n * n


# -- raster --------------------------------------------------------------------------


def _grid(n=6, levels=("a", "b", "c")):
    xx, yy = np.meshgrid(np.arange(n), np.arange(n))
    v = np.array(levels)[(xx + 2 * yy).ravel() % len(levels)]
    return pd.DataFrame({"x": xx.ravel(), "y": yy.ravel(), "v": v})


def test_raster_one_mobject_per_colour():
    df = _grid()
    g = ggplot(df, aes("x", "y", fill="v")).add(geom_raster()).build()
    mob = g.layers[0]
    assert len(mob.marks) == 3
    assert all(isinstance(k, str) and k.startswith("#") for k in mob.keys)
    # every tile is a 4-corner subpath of its colour's mobject
    assert sum(len(m.get_subpaths()) for m in mob.marks) == len(df)


def test_raster_irregular_falls_back_to_tiles():
    df = pd.DataFrame({"x": [0, 1, 2.5], "y": [0, 0, 0], "v": ["a", "a", "b"]})
    mob = ggplot(df, aes("x", "y", fill="v")).add(geom_raster()).build().layers[0]
    assert len(mob.marks) == 3 and mob.keys == [0, 1, 2]


def test_raster_hjust_vjust():
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    out = GeomRaster().setup_data(df, {"hjust": 0.0, "vjust": 1.0})
    assert list(out["xmin"]) == [-1, 0] and list(out["xmax"]) == [0, 1]
    assert list(out["ymin"]) == [0, 1] and list(out["ymax"]) == [1, 2]


def _pixels(mob) -> np.ndarray:
    cam = Camera(pixel_width=320, pixel_height=180)
    cam.capture_mobject(mob)
    return cam.pixel_array.astype(float)


def test_raster_renders_like_tile():
    df = _grid(8)
    colours = scale_fill_manual(values={"a": "#E41A1C", "b": "#377EB8", "c": "#4DAF4A"})
    small = theme(panel_width=4, panel_height=3)
    tile = ggplot(df, aes("x", "y", fill="v")).add(geom_tile(), colours, small).build()
    raster = ggplot(df, aes("x", "y", fill="v")).add(geom_raster(), colours, small).build()
    assert _close(tile.layers[0].get_center(), raster.layers[0].get_center())
    a, b = _pixels(tile.layers[0]), _pixels(raster.layers[0])
    covered = (a[..., :3].max(axis=2) > 0).sum()
    assert covered > 0.05 * a.shape[0] * a.shape[1]  # the tiles are in the frame
    diff = np.abs(a - b).max(axis=2)
    # pixels may differ only on the boundaries between tiles of different colours
    # (where the anti-aliased edges are painted in a different order)
    edge = np.zeros(diff.shape, dtype=bool)
    for axis in (0, 1):
        step = np.abs(np.diff(a, axis=axis)).max(axis=2) > 2
        lo = [slice(None)] * 2
        hi = [slice(None)] * 2
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        edge[tuple(lo)] |= step
        edge[tuple(hi)] |= step
    grown = edge.copy()
    for shift in (-1, 1):
        grown |= np.roll(edge, shift, 0) | np.roll(edge, shift, 1)
    assert not (diff > 2)[~grown].any()
    assert (diff > 2).sum() < 0.25 * covered


def test_raster_under_coord_flip():
    df = _grid(4)
    g = ggplot(df, aes("x", "y", fill="v")).add(geom_raster(), coord_flip()).build()
    m = g.layers[0].marks[0]
    first = np.asarray(m.get_subpaths()[0])[::4][:4]
    data = g.layer_data[0].iloc[0]
    want = g.panel.transform(
        [data.xmin, data.xmax, data.xmax, data.xmin], [data.ymin, data.ymin, data.ymax, data.ymax]
    )
    assert _close(first, want)


def test_tile_discrete_scale_object():
    df = pd.DataFrame({"x": ["a", "b"], "y": [1.0, 2.0]})
    g = ggplot(df, aes("x", "y")).add(geom_tile(), scale_x_discrete()).build()
    assert len(g.layers[0].marks) == 2


# -- shared helpers ------------------------------------------------------------------


def test_ring_points_are_straight_closed_curves():
    ring = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=float)
    pts = ring_points(ring)
    assert pts.shape == (12, 3)
    assert _close(pts[-1], ring[0])  # closed
    assert _close(pts[1], [1 / 3, 0, 0]) and _close(pts[2], [2 / 3, 0, 0])
    m = rings_mobject([ring, ring + 5])
    assert len(m.get_subpaths()) == 2
