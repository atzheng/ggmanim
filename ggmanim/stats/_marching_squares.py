"""Marching squares (numpy): iso-lines and iso-bands of a gridded field.

The field is ``z[j, i]`` at ``(x[i], y[j])`` on a rectilinear grid (``x`` and
``y`` ascending). A corner is *inside* at level ``L`` when ``z >= L``.

* Each cell gets one of 16 cases from its four corners (bits: bottom-left 1,
  bottom-right 2, top-right 4, top-left 8). The two saddle cases (5 and 10) are
  disambiguated by the cell average (``>= L``: the inside corners connect through
  the centre).
* Crossing points are linearly interpolated along the cell edges.
* Every segment is oriented with the inside on its *left*, so the joined
  polylines run counter-clockwise around superlevel regions (and clockwise around
  holes). Segments are joined through the grid edge they share (each edge carries
  at most one crossing, entered by one cell and left by the neighbour), which is
  exact -- no floating-point endpoint hashing is needed.

:func:`isolines` returns open or closed polylines at one level. :func:`isoband`
returns the closed, oriented rings of the band ``lo <= z < hi``: the boundary of
``{z >= lo}`` plus the reversed boundary of ``{z >= hi}`` (the field is padded
with a degenerate rim below every level, so all rings close along the grid
border). Filled with the *nonzero* rule, those rings cover exactly the band.

Pure numpy; no manim imports.
"""

from __future__ import annotations

import numpy as np

__all__ = ["grid_from_xyz", "isolines", "isoband", "SEGMENT_TABLE"]

# edges: 0 = bottom, 1 = right, 2 = top, 3 = left
#: case -> oriented segments ``(from_edge, to_edge)``; saddles (5, 10) are listed
#: as ``(centre outside, centre inside)``.
SEGMENT_TABLE: dict[int, tuple] = {
    0: (),
    1: ((0, 3),),
    2: ((1, 0),),
    3: ((1, 3),),
    4: ((2, 1),),
    5: (((0, 3), (2, 1)), ((0, 1), (2, 3))),
    6: ((2, 0),),
    7: ((2, 3),),
    8: ((3, 2),),
    9: ((0, 2),),
    10: (((1, 0), (3, 2)), ((3, 0), (1, 2))),
    11: ((1, 2),),
    12: ((3, 1),),
    13: ((0, 1),),
    14: ((3, 0),),
    15: (),
}


def grid_from_xyz(x, y, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Long ``(x, y, z)`` data -> ``(ux, uy, Z)`` with ``Z[j, i]`` at ``(ux[i], uy[j])``.

    ``ux``/``uy`` are the sorted unique coordinates; grid cells without a value
    are NaN (like ggplot2's ``xyz_to_isolines``)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    ux, xi = np.unique(x, return_inverse=True)
    uy, yi = np.unique(y, return_inverse=True)
    Z = np.full((len(uy), len(ux)), np.nan)
    Z[yi, xi] = z
    return ux, uy, Z


def _edge_ids(ny: int, nx: int, j: np.ndarray, i: np.ndarray, edge: int) -> np.ndarray:
    """Global id of edge ``edge`` of cells ``(j, i)``: horizontal edges first
    (``ny * (nx - 1)``), then vertical ones."""
    nh = ny * (nx - 1)
    if edge == 0:
        return j * (nx - 1) + i
    if edge == 2:
        return (j + 1) * (nx - 1) + i
    if edge == 3:
        return nh + j * nx + i
    return nh + j * nx + i + 1  # right


def _edge_point(x, y, Z, level: float, eid: int) -> tuple[float, float]:
    ny, nx = Z.shape
    nh = ny * (nx - 1)
    if eid < nh:  # horizontal: (j, i) -> (j, i + 1)
        j, i = divmod(eid, nx - 1)
        z0, z1 = Z[j, i], Z[j, i + 1]
        t = (level - z0) / (z1 - z0) if z1 != z0 else 0.5
        return x[i] + t * (x[i + 1] - x[i]), y[j]
    j, i = divmod(eid - nh, nx)  # vertical: (j, i) -> (j + 1, i)
    z0, z1 = Z[j, i], Z[j + 1, i]
    t = (level - z0) / (z1 - z0) if z1 != z0 else 0.5
    return x[i], y[j] + t * (y[j + 1] - y[j])


def _segments(Z: np.ndarray, level: float) -> list[tuple[int, int]]:
    """Oriented segments ``(from_edge_id, to_edge_id)`` of the level set."""
    ny, nx = Z.shape
    if ny < 2 or nx < 2:
        return []
    inside = Z >= level
    bl, br = inside[:-1, :-1], inside[:-1, 1:]
    tr, tl = inside[1:, 1:], inside[1:, :-1]
    case = bl * 1 + br * 2 + tr * 4 + tl * 8
    corners = np.stack([Z[:-1, :-1], Z[:-1, 1:], Z[1:, 1:], Z[1:, :-1]])
    valid = np.isfinite(corners).all(axis=0)
    case = np.where(valid, case, 0)
    with np.errstate(invalid="ignore"):
        centre_in = corners.mean(axis=0) >= level
    segs: list[tuple[int, int]] = []
    for c in range(1, 15):
        jj, ii = np.nonzero(case == c)
        if not len(jj):
            continue
        entries = SEGMENT_TABLE[c]
        if c in (5, 10):
            ci = centre_in[jj, ii]
            for mask, pairs in ((~ci, entries[0]), (ci, entries[1])):
                for a, b in pairs:
                    ea = _edge_ids(ny, nx, jj[mask], ii[mask], a)
                    eb = _edge_ids(ny, nx, jj[mask], ii[mask], b)
                    segs.extend(zip(ea.tolist(), eb.tolist()))
            continue
        for a, b in entries:
            ea = _edge_ids(ny, nx, jj, ii, a)
            eb = _edge_ids(ny, nx, jj, ii, b)
            segs.extend(zip(ea.tolist(), eb.tolist()))
    return segs


def _chains(segs: list[tuple[int, int]]) -> list[tuple[list[int], bool]]:
    """Join oriented segments into edge-id chains: ``(ids, closed)``. Open chains
    start at an edge no segment leads to; the rest are cycles (first id not
    repeated)."""
    nxt = dict(segs)
    targets = set(nxt.values())
    out: list[tuple[list[int], bool]] = []
    seen: set[int] = set()
    # open chains first, in a deterministic order
    for start in sorted(k for k in nxt if k not in targets):
        chain = [start]
        seen.add(start)
        cur = start
        while cur in nxt and nxt[cur] not in seen:
            cur = nxt[cur]
            chain.append(cur)
            seen.add(cur)
        out.append((chain, False))
    for start in sorted(nxt):
        if start in seen:
            continue
        chain = [start]
        seen.add(start)
        cur = nxt[start]
        while cur != start and cur not in seen:
            chain.append(cur)
            seen.add(cur)
            cur = nxt.get(cur, start)
        out.append((chain, True))
    return out


def _coords(x, y, Z, level, ids: list[int]) -> np.ndarray:
    pts = np.array([_edge_point(x, y, Z, level, e) for e in ids], dtype=float)
    if len(pts) > 1:  # drop repeated vertices (crossings exactly on a grid node)
        keep = np.ones(len(pts), dtype=bool)
        keep[1:] = np.any(np.abs(np.diff(pts, axis=0)) > 1e-12, axis=1)
        pts = pts[keep]
    return pts


def isolines(x, y, Z, level: float) -> list[np.ndarray]:
    """Polylines (``(k, 2)`` arrays) where the grid field crosses ``level``.

    Closed lines repeat their first vertex at the end. Cells with a missing
    corner are skipped, so contours stop there (and at the grid border)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    out = []
    for ids, closed in _chains(_segments(Z, level)):
        pts = _coords(x, y, Z, level, ids)
        if closed and len(pts) >= 3:
            pts = np.vstack([pts, pts[:1]])
        if len(pts) >= 2:
            out.append(pts)
    return out


def _rings(x, y, Z, level: float) -> list[np.ndarray]:
    """Closed, oriented boundary rings (no repeated vertex) of ``{Z >= level}``;
    ``Z`` must be padded so that every chain closes."""
    out = []
    for ids, _closed in _chains(_segments(Z, level)):
        pts = _coords(x, y, Z, level, ids)
        if len(pts) > 1 and np.all(np.abs(pts[0] - pts[-1]) <= 1e-12):
            pts = pts[:-1]
        if len(pts) >= 3:
            out.append(pts)
    return out


def _pad(x, y, Z, low: float):
    """Add a degenerate rim (duplicated border coordinates) valued ``low``."""
    xp = np.concatenate([[x[0]], x, [x[-1]]])
    yp = np.concatenate([[y[0]], y, [y[-1]]])
    Zp = np.full((len(y) + 2, len(x) + 2), low)
    Zp[1:-1, 1:-1] = np.where(np.isfinite(Z), Z, low)
    return xp, yp, Zp


def isoband(x, y, Z, lo: float, hi: float) -> list[np.ndarray]:
    """Closed rings (``(k, 2)`` arrays, last vertex not repeated) of the band
    ``lo <= Z < hi``: counter-clockwise rings bound ``{Z >= lo}`` (holes
    clockwise), and the boundary of ``{Z >= hi}`` is added reversed. Filled with
    the nonzero (or even-odd) rule they cover the band. Missing values count as
    below every level."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    finite = Z[np.isfinite(Z)]
    if not len(finite) or len(x) < 2 or len(y) < 2:
        return []
    low = min(float(finite.min()), lo, hi) - 1.0
    xp, yp, Zp = _pad(x, y, Z, low)
    rings = _rings(xp, yp, Zp, lo)
    if not rings:
        return []
    upper = [r[::-1] for r in _rings(xp, yp, Zp, hi)]
    # a ring of {Z >= hi} that is also a ring of {Z >= lo} (e.g. the grid border
    # when the whole border is above hi) cancels out: drop both copies
    keys = {_ring_key(r): k for k, r in enumerate(rings)}
    drop_lo, keep_up = set(), []
    for r in upper:
        k = keys.get(_ring_key(r))
        if k is not None and k not in drop_lo:
            drop_lo.add(k)
        else:
            keep_up.append(r)
    return [r for k, r in enumerate(rings) if k not in drop_lo] + keep_up


def _ring_key(ring: np.ndarray) -> tuple:
    """Orientation- and start-independent key of a ring's vertex set."""
    pts = np.round(np.asarray(ring, dtype=float), 9) + 0.0
    return tuple(sorted(map(tuple, pts.tolist())))
