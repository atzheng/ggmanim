"""``coord_trans``: transformed cartesian coordinates (ggplot2's ``coord_trans``).

The transform is applied by the coord at drawing time, not by the scales::

    ggplot(df, aes("x", "y")).geom_line().add(coord_trans(x="log10"))

* Stats and position scales work on untransformed data (an ``lm`` smooth is a
  straight line in data space), limits and breaks come from the scales.
* :meth:`CoordTrans.build` returns a :class:`TransPanel`. Its manim axes run over
  the transformed, expanded ranges, so ticks and grid lines sit at the
  transformed break positions. Its ``transform(x, y)`` and ``panel_params`` are
  in the coord's *input* space, so geoms clip as usual and pass untransformed
  values to ``transform``.
* ``TransPanel.is_linear`` is false: geoms munch paths, polygons and rects
  (:func:`ggmanim.coords.coord_munch` via ``geoms.base.munch_path``) so straight
  segments bend with the transform.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .coords import _FALLBACK_RANGE, CartesianPanel, CoordCartesian, PanelParams, _c2p
from .registry import register
from .scales.breaks import expand_range
from .scales.continuous import get_transform
from .theme import Theme

__all__ = ["TransPanel", "TransPanelParams", "CoordTrans", "coord_trans"]


class TransPanel(CartesianPanel):
    """A :class:`CartesianPanel` for :func:`coord_trans`.

    The manim axes run over the *coord-transformed* space (``trans_x(x)``,
    ``trans_y(y)``), but everything geoms see is in the coord's *input* space (the
    position scales' output, usually data space): ``transform(x, y)`` applies the
    transforms, then ``axes.c2p``; ``panel_params`` (ranges, breaks, minor breaks)
    are input-space values, so clipping against ``x_range``/``y_range`` happens
    before the (monotone) transform. ``trans_range`` holds the axes'
    transformed-space ``(x_range, y_range)``.
    """

    is_linear = False
    trans_x: Any = None
    trans_y: Any = None
    trans_range: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0))

    def transform(self, x: Any, y: Any) -> np.ndarray:
        """Input-space ``(x, y)`` -> ``(N, 3)`` scene points.

        As in ggplot2 (``transform_value`` + ``squish_infinite``), a value whose
        image is infinite -- ``+-inf``, or the open end of a transform's domain
        such as ``0`` under ``log`` -- goes to that edge of the panel, so a bar
        from 0 on a log axis reaches the panel edge even with ``clip="off"``.
        Other out-of-domain values (``log`` of a negative) map to NaN."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        (rx, ry) = self.trans_range
        return _c2p(self.axes, _squished(self.trans_x, x, rx), _squished(self.trans_y, y, ry))

    def squish_input(self, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(x, y)`` with each value drawn at a panel edge by :meth:`transform`
        (infinite image) replaced by the input-space value of that edge.
        :func:`~ggmanim.coords.coord_munch` calls it first (ggplot2 does the same
        with ``+-Inf``), so a path from ``0`` on a log axis is interpolated from
        the panel edge instead of through ever smaller values below it."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        (rx, ry) = self.trans_range
        return _edge_input(self.trans_x, x, rx), _edge_input(self.trans_y, y, ry)

    def munch_distance(self, x: Any, y: Any) -> np.ndarray:
        """Segment lengths of the path ``(x, y)`` in transformed space, each axis
        divided by its transformed range (ggplot2's ``dist_euclidean`` on
        rescaled values)."""
        (x0, x1), (y0, y1) = self.trans_range
        tx = _squished(self.trans_x, x, (x0, x1)) / ((x1 - x0) or 1.0)
        ty = _squished(self.trans_y, y, (y0, y1)) / ((y1 - y0) or 1.0)
        return np.hypot(np.diff(tx), np.diff(ty))


@dataclass(frozen=True)
class TransPanelParams(PanelParams):
    """:class:`PanelParams` of a :class:`TransPanel`: the inherited fields are in
    the coord's input space (what geoms clip against); ``trans_x_range`` /
    ``trans_y_range`` are the drawn, transformed-space axis ranges (what
    ``facet_grid(space="free")`` sizes panels by)."""

    trans_x_range: tuple[float, float] | None = None
    trans_y_range: tuple[float, float] | None = None


def _edge_input(trans: Any, v: np.ndarray, rng: tuple[float, float]) -> np.ndarray:
    """``v`` with the values :func:`_squished` puts at an end of ``rng`` replaced
    by that end's input-space value."""
    with np.errstate(all="ignore"):
        raw = np.asarray(trans.transform(v), dtype=float)
    if np.isfinite(raw).all():
        return v.copy()
    t = _squished(trans, v, rng)
    edge = ~np.isfinite(raw) & np.isfinite(t)
    with np.errstate(all="ignore"):
        back = np.asarray(trans.inverse(t), dtype=float)
    return np.where(edge, back, v)


def _squished(trans: Any, v: Any, rng: tuple[float, float]) -> np.ndarray:
    """``trans.transform(v)`` with infinite images squished into ``rng``: ``+-inf``
    inputs whose image is infinite, and values just outside an *open* domain end
    (``0`` for ``log``, whose domain starts at the smallest positive float), go
    to the end of ``rng`` that end of the domain maps towards."""
    v = np.asarray(v, dtype=float)
    with np.errstate(all="ignore"):
        t = np.array(trans.transform(v), dtype=float, copy=True)
    if np.isfinite(t).all():
        return t
    lo, hi = sorted(float(e) for e in rng)
    t = np.where(t == np.inf, hi, np.where(t == -np.inf, lo, t))
    dlo, dhi = (float(d) for d in trans.domain)
    for bound, step, at_edge in (
        (dlo, 1.0, (v < dlo) & (np.nextafter(v, np.inf) >= dlo)),
        (dhi, -1.0, (v > dhi) & (np.nextafter(v, -np.inf) <= dhi)),
    ):
        if not (np.isfinite(bound) and at_edge.any()):
            continue
        inside = bound + step * max(1.0, abs(bound))  # a point well inside the domain
        with np.errstate(all="ignore"):
            image, ref = (float(i) for i in np.asarray(trans.transform([bound, inside]), float))
        if np.isfinite(image) and np.isfinite(ref):
            t = np.where(at_edge, lo if image < ref else hi, t)
    return t


def _inverse_range(trans: Any, rng: tuple[float, float]) -> tuple[float, float]:
    """Input-space ``(lo, hi)`` whose image is the transformed range ``rng``. An
    end whose inverse does not map back (``sqrt`` expanded below 0) is clamped to
    the domain bound whose image is nearest."""
    t = np.asarray(rng, dtype=float)
    with np.errstate(all="ignore"):
        d = np.asarray(trans.inverse(t), dtype=float)
        back = np.asarray(trans.transform(d), dtype=float)
        dom = np.asarray(trans.domain, dtype=float)
        tdom = np.asarray(trans.transform(dom), dtype=float)
    for i in range(2):
        if not (np.isfinite(back[i]) and abs(back[i] - t[i]) <= 1e-9 * max(1.0, abs(t[i]))):
            gap = np.where(np.isfinite(tdom), np.abs(tdom - t[i]), np.inf)
            if np.isfinite(gap).any():
                d[i] = dom[int(np.argmin(gap))]
    lo, hi = sorted(float(v) for v in d)
    return lo, hi


class _TransScale:
    """A position-scale facade in coord-transformed space, so that
    :meth:`CoordCartesian.build` draws the axes, grid and ticks of a
    :class:`CoordTrans` panel unchanged: breaks/labels come from the real scale
    (input space, over :attr:`input_range`) and are mapped through ``trans``.
    Other attributes (``name``, ``position`` ...) pass through to the scale."""

    def __init__(self, scale: Any, trans: Any, trans_range: tuple[float, float]):
        self._scale = scale
        self._trans = trans
        self._range = trans_range
        self.input_range = _inverse_range(trans, trans_range)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.__dict__.get("_scale"), name)

    def get_expanded_limits(self, expand: Any = None) -> tuple[float, float]:
        return self._range

    def pairs(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """``(input-space breaks, transformed breaks, labels)`` inside the panel."""
        if self._scale is None:
            return np.zeros(0), np.zeros(0), []
        b = np.asarray(self._scale.get_breaks(self.input_range), dtype=float).ravel()
        labels = [str(s) for s in self._scale.get_labels(self.input_range)]
        if len(labels) != len(b):
            raise ValueError(f"scale {self._scale!r}: {len(labels)} labels for {len(b)} breaks")
        with np.errstate(all="ignore"):
            t = np.asarray(self._trans.transform(b), dtype=float)
        lo, hi = self._range
        tol = 1e-9 * (hi - lo)
        keep = np.isfinite(t) & (t >= lo - tol) & (t <= hi + tol)
        return b[keep], t[keep], [s for s, k in zip(labels, keep) if k]

    def get_breaks(self, limits: Any = None) -> np.ndarray:
        return self.pairs()[1]

    def get_labels(self, limits: Any = None) -> list[str]:
        return self.pairs()[2]

    def minor_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """``(input-space, transformed)`` minor breaks with a finite image."""
        get = getattr(self._scale, "get_minor_breaks", None)
        if get is None:
            return np.zeros(0), np.zeros(0)
        b = np.asarray(get(self.input_range), dtype=float).ravel()
        with np.errstate(all="ignore"):
            t = np.asarray(self._trans.transform(b), dtype=float)
        keep = np.isfinite(t)
        return b[keep], t[keep]

    def get_minor_breaks(self, limits: Any = None) -> np.ndarray:
        return self.minor_pairs()[1]


class CoordTrans(CoordCartesian):
    """Transformed cartesian coordinates (ggplot2's ``coord_trans``).

    Unlike a scale transform (``scale_x_log10``), the transform applies at drawing
    time, *after* the stats: stats and position scales see untransformed data (a
    ``geom_smooth(method="lm")`` fits a straight line in data space, drawn
    curved), breaks and labels are the scale's own (data-space ``pretty`` breaks
    for a continuous scale) placed at their transformed positions, and the
    straight edges of paths, polygons and rects are munched (:func:`~ggmanim.coords.coord_munch`)
    so they bend with the transform.

    ``x``/``y`` are transform names (``"identity"``, ``"log10"``, ``"log2"``,
    ``"log"``, ``"sqrt"``, ``"reverse"``) or
    :class:`~ggmanim.scales.continuous.Transform` instances. ``xlim``/``ylim``
    zoom in data space (as in :func:`coord_cartesian`); ``expand`` pads the
    *transformed* range. A value at the open end of a transform's domain
    (``log`` of 0) or an infinite one is drawn at the panel edge, as in ggplot2,
    so bars from 0 reach the bottom of a log axis also with ``clip="off"``;
    other out-of-domain values (``log`` of a negative) map to NaN and are not
    drawn.

    :meth:`build` returns a :class:`TransPanel` (``is_linear`` false).
    """

    is_linear = False

    def __init__(
        self,
        x: Any = "identity",
        y: Any = "identity",
        xlim: Sequence[float | None] | None = None,
        ylim: Sequence[float | None] | None = None,
        expand: bool = True,
        clip: str = "on",
    ):
        super().__init__(xlim=xlim, ylim=ylim, expand=expand, clip=clip)
        self.trans_x = get_transform(x)
        self.trans_y = get_transform(y)

    def __repr__(self):
        clip = f", clip={self.clip!r}" if self.clip != "on" else ""
        return (
            f"coord_trans(x={self.trans_x.name!r}, y={self.trans_y.name!r}, "
            f"xlim={self.xlim!r}, ylim={self.ylim!r}, expand={self.expand!r}{clip})"
        )

    def _trans_range(self, scale: Any, lim: Any, trans: Any) -> tuple[float, float]:
        """Transformed-space axis range: the scale's input-space limits (``lim``
        zooms), mapped through ``trans``, then expanded."""
        if scale is None:
            return _FALLBACK_RANGE
        discrete = hasattr(scale, "continuous_limits")
        known = scale.get_expanded_limits(False) if discrete else scale.get_limits()
        known = known if known is not None else _FALLBACK_RANGE
        ends = [float(min(known)), float(max(known))]
        if lim is not None:
            inc = scale._is_increasing() if hasattr(scale, "_is_increasing") else True
            t = np.asarray(scale.transform([np.nan if v is None else v for v in lim]), dtype=float)
            ends = [
                t[0] if np.isfinite(t[0]) else ends[0 if inc else 1],
                t[1] if np.isfinite(t[1]) else ends[1 if inc else 0],
            ]
        with np.errstate(all="ignore"):
            te = np.asarray(trans.transform(np.asarray(ends, dtype=float)), dtype=float)
        if not np.isfinite(te).all():  # e.g. log of a range starting at 0
            fin = te[np.isfinite(te)]
            te = np.array([fin[0], fin[0]]) if len(fin) else np.array(_FALLBACK_RANGE)
        expand: Any = False
        if self.expand:
            expand = getattr(scale, "expand", None)
            if expand is None and discrete:
                expand = (0.0, 0.6)
        lo, hi = expand_range(tuple(sorted(float(v) for v in te)), expand)
        if hi <= lo:
            lo, hi = lo - 0.5, hi + 0.5
        return lo, hi

    def _facades(self, x_scale: Any, y_scale: Any) -> tuple[_TransScale, _TransScale]:
        tx, ty = self.trans_x, self.trans_y
        return (
            _TransScale(x_scale, tx, self._trans_range(x_scale, self.xlim, tx)),
            _TransScale(y_scale, ty, self._trans_range(y_scale, self.ylim, ty)),
        )

    def setup_panel_params(self, x_scale, y_scale) -> PanelParams:
        """Input-space panel params (what geoms see): the ranges are the inverse
        images of the transformed axis ranges, the breaks the scale's breaks inside
        them."""
        fx, fy = self._facades(x_scale, y_scale)
        (xb, _, xl), (yb, _, yl) = fx.pairs(), fy.pairs()

        def minor(f: _TransScale) -> tuple[float, ...]:
            return tuple(float(v) for v in f.minor_pairs()[0])

        return TransPanelParams(
            trans_x_range=fx._range,
            trans_y_range=fy._range,
            x_range=fx.input_range,
            y_range=fy.input_range,
            x_breaks=tuple(float(v) for v in xb),
            y_breaks=tuple(float(v) for v in yb),
            x_labels=tuple(xl),
            y_labels=tuple(yl),
            x_minor_breaks=minor(fx),
            y_minor_breaks=minor(fy),
        )

    def build(
        self,
        x_scale,
        y_scale,
        theme: Theme | None = None,
        labels: Mapping[str, Any] | None = None,
        *,
        has_legend: bool = False,
    ) -> TransPanel:
        """Build a :class:`TransPanel`: a cartesian panel over the transformed ranges
        (grid lines and ticks at the transformed breaks) whose ``transform`` and
        ``panel_params`` work in input space."""
        fx, fy = self._facades(x_scale, y_scale)
        inner = CoordCartesian(expand=self.expand, clip=self.clip)
        panel = inner.build(fx, fy, theme, labels, has_legend=has_legend)
        panel.__class__ = TransPanel  # same state; transform/munching become non-linear
        assert isinstance(panel, TransPanel)
        panel.trans_x, panel.trans_y = self.trans_x, self.trans_y
        panel.trans_range = (fx._range, fy._range)
        panel.panel_params = self.setup_panel_params(x_scale, y_scale)
        return panel


@register
def coord_trans(
    x: Any = "identity",
    y: Any = "identity",
    xlim=None,
    ylim=None,
    expand: bool = True,
    clip: str = "on",
) -> CoordTrans:
    """Transformed coordinates: ``coord_trans(x="log10", y="sqrt")`` transforms the
    axes at drawing time, after the stats (see :class:`CoordTrans`); straight lines
    become curves."""
    return CoordTrans(x=x, y=y, xlim=xlim, ylim=ylim, expand=expand, clip=clip)
