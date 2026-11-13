"""Dashing for the ``linetype`` aesthetic.

:func:`apply_linetype` turns a stroked path into a ``VGroup`` of dash segments.
Patterns come from :func:`ggmanim.scales.shape.parse_linetype` and are, as in
ggplot2, alternating on/off lengths in units of the line width: one unit is
:data:`DASH_UNIT` scene units per manim stroke-width unit, so a ``linewidth=3``
``"dashed"`` line has 0.12-unit dashes and gaps whatever the path's length or
the number of vertices (unlike ``DashedVMobject``'s ``num_dashes``, which is
relative to the path). Dashes follow the path through its corners; the pattern
is measured by arc length along the whole path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from manim import VGroup, VMobject

from ..scales.shape import parse_linetype

__all__ = ["DASH_UNIT", "apply_linetype", "dash_polyline"]

#: Scene units of one dash-pattern unit per unit of manim stroke width.
DASH_UNIT = 0.01

#: Bezier samples per curved segment when flattening a path.
_CURVE_SAMPLES = 12


def _flatten(mob: VMobject) -> list[np.ndarray]:
    """The subpaths of ``mob`` as polylines (straight segments kept exact,
    curved cubic segments sampled)."""
    out = []
    for sub in mob.get_subpaths():
        sub = np.asarray(sub, dtype=float)
        if len(sub) < 4:
            continue
        pts = [sub[0]]
        for i in range(0, len(sub) - 3, 4):
            a, h1, h2, b = sub[i : i + 4]
            chord = b - a
            n = float(np.linalg.norm(chord))
            if n == 0:
                straight = np.allclose(h1, a) and np.allclose(h2, a)
            else:
                d1 = np.linalg.norm(np.cross(h1 - a, chord)) / n
                d2 = np.linalg.norm(np.cross(h2 - a, chord)) / n
                straight = max(d1, d2) < 1e-9 * max(n, 1.0)
            if straight:
                pts.append(b)
            else:
                t = np.linspace(0, 1, _CURVE_SAMPLES + 1)[1:, None]
                pts.extend(
                    (1 - t) ** 3 * a
                    + 3 * (1 - t) ** 2 * t * h1
                    + 3 * (1 - t) * t**2 * h2
                    + t**3 * b
                )
        out.append(np.asarray(pts))
    return out


def dash_polyline(points: np.ndarray, pattern: tuple[float, ...]) -> list[np.ndarray]:
    """Cut the polyline ``points`` (N, 3) into the "on" pieces of ``pattern``
    (on/off lengths in scene units, repeated; must be positive)."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2 or not pattern:
        return []
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return []

    def at(u: float) -> np.ndarray:
        return np.array([np.interp(u, s, points[:, k]) for k in range(points.shape[1])])

    dashes = []
    t, i = 0.0, 0
    period = float(sum(pattern))
    max_pieces = int(total / period * len(pattern)) + len(pattern) + 2
    while t < total and i < max_pieces:
        length = pattern[i % len(pattern)]
        if i % 2 == 0:
            a, b = t, min(t + length, total)
            inner = points[(s > a) & (s < b)]
            dashes.append(np.vstack([at(a), inner, at(b)]))
        t += length
        i += 1
    return dashes


def apply_linetype(mob: VMobject, linetype: Any, *, max_period: float | None = None) -> VMobject:
    """``mob`` drawn with ``linetype``.

    Solid (``None``/``"solid"``) returns ``mob`` itself; ``"blank"`` returns it
    with zero stroke opacity; otherwise a ``VGroup`` of dash ``VMobject``\\ s with
    ``mob``'s stroke colour, width and opacity. The pattern unit is
    ``DASH_UNIT * stroke_width``; ``max_period`` (scene units) shrinks it so one
    full pattern repeat is at most that long (used for legend keys).
    Raises ``ValueError`` for unknown linetypes.
    """
    pattern = parse_linetype(linetype)
    if pattern is None:
        return mob
    if len(pattern) == 0:
        mob.set_stroke(opacity=0)
        return mob
    width = float(mob.get_stroke_width())
    unit = DASH_UNIT * max(width, 0.5)
    if max_period is not None and unit * sum(pattern) > max_period:
        unit = max_period / sum(pattern)
    scaled = tuple(p * unit for p in pattern)
    color: Any = mob.get_stroke_color()
    opacity = mob.get_stroke_opacity()
    out: Any = VGroup()
    for poly in _flatten(mob):
        for dash in dash_polyline(poly, scaled):
            d = VMobject()
            d.set_points_as_corners(dash)
            d.set_fill(opacity=0)
            d.set_stroke(color=color, width=width, opacity=opacity)
            out.add(d)
    out.linetype = linetype
    return out
