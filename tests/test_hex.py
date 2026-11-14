"""stat_bin_hex / geom_hex: hexbin lattice, nearest-centre counts, tessellation."""

import numpy as np
import pandas as pd
import pytest

from ggmanim import aes, coord_flip, geom_hex, ggplot, stat_bin_hex
from ggmanim.geoms import GeomHex
from ggmanim.registry import COMPONENTS
from ggmanim.stats import StatBinHex, hexagon_vertices, hexbin

SQRT3 = np.sqrt(3)


def cloud(n=800, seed=2):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.normal(size=n), "y": 2 * rng.normal(size=n)})


def test_registered():
    for n in ("geom_hex", "stat_bin_hex", "stat_binhex"):
        assert n in COMPONENTS
    assert isinstance(stat_bin_hex().geom, GeomHex)


def test_counts_sum_to_n_and_columns():
    out = StatBinHex(bins=12).compute(cloud())
    assert out["count"].sum() == 800
    assert {"x", "y", "width", "height", "count", "ncount", "density", "ndensity"} <= set(
        out.columns
    )
    assert out["density"].sum() == pytest.approx(1.0)


def test_nearest_centre_assignment():
    df = cloud(300)
    wx, wy = 0.4, 0.9
    out = hexbin(df["x"], df["y"], (wx, wy))
    # a point's nearest lattice centre always holds it, so brute force over the
    # returned centres (in bin-width units) must give the same counts
    cx, cy = out["x"].to_numpy() / wx, out["y"].to_numpy() / wy
    px, py = df["x"].to_numpy() / wx, df["y"].to_numpy() / wy
    d = (px[:, None] - cx[None, :]) ** 2 + (py[:, None] - cy[None, :]) ** 2
    counts = np.bincount(d.argmin(axis=1), minlength=len(out))
    assert counts.tolist() == out["count"].astype(int).tolist()
    # centres on the two staggered lattices: x steps of wx/2, rows sqrt(3)/2 * wy apart
    rows = np.round((cy - cy.min()) / (SQRT3 / 2), 6)
    assert np.allclose(rows, np.round(rows))


def test_lattice_is_regular_hexagonal():
    out = hexbin(*cloud(3000).to_numpy().T, binwidth=(1.0, 1.0))
    c = out[["x", "y"]].to_numpy()
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    assert np.allclose(d.min(axis=1)[d.min(axis=1) < 1.5], 1.0)  # nearest neighbours 1 apart


def test_hexagons_tessellate():
    # neighbouring hexagons share edges exactly: every interior edge appears twice
    centres = hexbin(*cloud(3000).to_numpy().T, binwidth=(0.5, 0.8))
    hx, hy = hexagon_vertices(centres["x"], centres["y"], 0.5, 0.8)
    edges = {}
    for vx, vy in zip(hx, hy):
        pts = [(round(a, 9), round(b, 9)) for a, b in zip(vx, vy)]
        for k in range(6):
            e = tuple(sorted((pts[k], pts[(k + 1) % 6])))
            edges[e] = edges.get(e, 0) + 1
    assert max(edges.values()) == 2
    shared = sum(1 for v in edges.values() if v == 2)
    assert shared > 2 * len(centres)  # dense interior: most edges shared
    # total hexagon area equals the lattice cell area (no overlap, no gap)
    area = 0.5 * np.abs((hx * np.roll(hy, -1, axis=1) - np.roll(hx, -1, axis=1) * hy).sum(axis=1))
    assert np.allclose(area, 0.5 * 0.8 * SQRT3 / 2)


def test_geom_draws_hexagons():
    g = ggplot(cloud(), aes("x", "y")).add(geom_hex(bins=10)).build()
    mob = g.layers[0]
    n = len(StatBinHex(bins=10).compute(cloud()))
    assert len(mob.marks) == n
    assert all(len(m.get_anchors()) // 2 >= 6 for m in mob.marks[:5])
    assert len(g.legends) == 1
    # scales cover whole hexagons
    stat = StatBinHex(bins=10).compute(cloud())
    lo = (stat["y"] - stat["height"] / SQRT3).min()
    assert min(g.panel.panel_params.y_range) <= lo + 1e-9


def test_geom_flip_and_params():
    g = ggplot(cloud(), aes("x", "y")).add(geom_hex(binwidth=(0.5, 1.0)), coord_flip()).build()
    assert len(g.layers[0].marks) > 10
    out = StatBinHex(binwidth=(0.5, 1.0)).compute(cloud())
    assert (out["width"] == 0.5).all() and (out["height"] == 1.0).all()
