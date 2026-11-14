"""The build pipeline: plot spec -> :class:`~ggmanim.mobjects.GGPlot`.

Mirrors ggplot2's ``ggplot_build`` + ``ggplot_gtable``, one function per step
(PLAN.md §2; the per-layer contract is spelled out in :mod:`ggmanim.geoms.base`):

1. :func:`resolve_layers` -- layer data/mapping (``inherit_aes``), evaluate the
   aesthetics, ``add_group``.
2. :func:`assign_panels` -- the facet's layout table (``plot.facet``, default
   :class:`~ggmanim.facets.FacetNull`) and the ``PANEL`` column of every layer
   (``facet.map_data``; see :mod:`ggmanim.facets.base` for the facet contract).
3. :func:`setup_position_scales` -- per-build position scales, *pre-stat* training
   on the raw layer data, then x-/y-like columns -> transformed space. With
   several panels :func:`setup_panel_scales` does this per panel
   (``facet.init_scales``: one shared scale when fixed, clones when free).
   :func:`compute_stats` -- ``stat.compute_layer`` per layer (per panel, with
   that panel's scales), then ``after_stat``.
4. :func:`map_non_position` -- train (on every layer) and map the non-position
   aesthetics: colour/fill/shape/linetype through discrete scales (numeric
   colour/fill through continuous colour gradients), size/alpha/linewidth through
   range scales.
5. :func:`setup_geom_data` -- ``geom.setup_data`` (e.g. bar extents).
6. :func:`apply_positions` + :func:`train_position_scales` -- each layer's
   :class:`~ggmanim.positions.Position` (``setup_params`` -> ``setup_data`` ->
   ``compute_layer``, per ``PANEL``; identity is a passthrough), then reset and
   retrain the position scales on what the geoms will draw (intercept-only
   layers contribute only their intercepts) and apply ``oob``. As in ggplot2,
   positions run *after* ``setup_data`` so they see bar extents
   (``xmin``/``xmax``/``ymin``/``ymax``).
7. :func:`build_panel` -- ``coord.build`` -> :class:`~ggmanim.coords.CartesianPanel`;
   with several panels :func:`build_panels` builds one per layout row and lays
   them out with strips (:class:`~ggmanim.facets.PanelGrid`).
8. :func:`draw_layers` -- ``geom.draw_panel`` per layer (per panel).
9. :func:`assemble` -- legends, titles, the :class:`~ggmanim.mobjects.GGPlot`.

:func:`build_plot` runs them all. It is pure: scales are cloned per build, layer
data is copied, user Mobjects (titles) are copied, and the spec is never mutated.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from manim import DOWN, LEFT, RIGHT, UP, Mobject, VGroup

# Importing these modules registers every component (geom_*, scale_*, ...).
from . import coords as _coords  # noqa: F401
from . import geoms as _geoms  # noqa: F401
from . import scales as _scales  # noqa: F401
from . import stats as _stats  # noqa: F401
from . import theme as _theme  # noqa: F401
from .aes import Aes, AfterStat
from .coords import CartesianPanel, CoordCartesian
from .facets import Facet, FacetNull, PanelGrid, PanelLayers
from .facets import _layout as _facet_layout
from .facets import base as _facet_base
from .guides import build_legends, place_legends  # also registers guides()
from .layer import Layer
from .positions import get_position
from .scales.continuous import X_AESTHETICS, Y_AESTHETICS, ScaleContinuousPosition
from .scales.date import ScaleDatePosition, data_timezone, date_kind
from .scales.discrete import CATEGORICAL_AESTHETICS, default_scale_for
from .scales.discrete_position import ScaleDiscretePosition
from .stats import add_group, is_discrete
from .stats.base import GROUPING_AES
from .theme import Theme, theme_default

__all__ = [
    "BuildError",
    "LayerData",
    "build_plot",
    "resolve_layers",
    "assign_panels",
    "setup_position_scales",
    "setup_panel_scales",
    "compute_stats",
    "map_non_position",
    "apply_positions",
    "setup_geom_data",
    "train_position_scales",
    "default_labels",
    "build_panel",
    "build_panels",
    "draw_layers",
    "assemble",
    "fit_to_frame",
]

POSITION_AES = {"x": X_AESTHETICS, "y": Y_AESTHETICS}


class BuildError(ValueError):
    """A plot spec that cannot be built (bad mapping, unsupported data ...)."""


@dataclass
class LayerData:
    """Working state of one layer during a build."""

    index: int
    layer: Layer
    mapping: Aes
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: the layer's resolved user data (facet variables are read from it)
    raw: pd.DataFrame | None = None
    #: ``{group id: raw grouping value}`` (a tuple when several aesthetics group)
    group_values: dict[int, Any] = field(default_factory=dict)

    @property
    def key(self):
        """Key of the layer in ``GGPlot.layers``: its name, else its index."""
        return self.layer.name if self.layer.name is not None else self.index

    @property
    def label(self) -> str:
        geom = type(self.layer.geom).__name__
        if self.layer.name is not None:
            return f"layer {self.layer.name!r} ({geom})"
        return f"layer {self.index} ({geom})"


# --------------------------------------------------------------------------- 1
def resolve_layers(plot) -> list[LayerData]:
    """Resolve each layer's data and mapping and evaluate its aesthetics.

    Returns one :class:`LayerData` per layer whose ``data`` has one column per
    (before-stat) aesthetic plus an integer ``group`` column. Unknown columns raise
    :class:`BuildError` naming the column, the aesthetic and the layer.
    """
    out = []
    for i, layer in enumerate(plot.layers):
        ld = LayerData(index=i, layer=layer, mapping=layer.resolve_mapping(plot.mapping))
        raw = layer.resolve_data(plot.data)
        if raw is not None and not isinstance(raw, pd.DataFrame):
            raise BuildError(
                f"{ld.label}: data must be a pandas DataFrame, got {type(raw).__name__}"
            )
        cols = list(raw.columns) if raw is not None else []
        for aes_name, value in ld.mapping.before_stat().items():
            if isinstance(value, str) and value not in cols:
                raise BuildError(
                    f"{ld.label}: aesthetic {aes_name!r} maps to column {value!r}, which "
                    f"is not in the layer data (columns: {cols})"
                )
        try:
            df = ld.mapping.evaluate(raw)
        except Exception as e:  # callables, length mismatches ...
            raise BuildError(f"{ld.label}: could not evaluate aesthetics: {e}") from e
        df = df.reset_index(drop=True)
        ld.data = add_group(df)
        ld.group_values = _group_values(df, ld.data["group"])
        ld.raw = raw
        out.append(ld)
    return out


def _scalar(v: Any) -> Any:
    return v.item() if isinstance(v, np.generic) else v


def _group_values(df: pd.DataFrame, group: pd.Series) -> dict[int, Any]:
    """``{group id: raw value}`` of the grouping aesthetics (an explicit ``group``,
    else the discrete :data:`~ggmanim.stats.base.GROUPING_AES`; a tuple when
    several are present) before scales map them."""
    if "group" in df.columns:
        cols = ["group"]
    else:
        cols = [a for a in GROUPING_AES if a in df.columns and is_discrete(df[a])]
    if not cols or not len(df):
        return {}
    firsts = df[cols].assign(_gid=group.to_numpy()).drop_duplicates("_gid")
    out: dict[int, Any] = {}
    for row in firsts.itertuples(index=False):
        vals = tuple(_scalar(v) for v in row[:-1])
        out[int(row[-1])] = vals[0] if len(vals) == 1 else vals
    return out


def _labels_by_group(label: Mapping, group_values: dict[int, Any]) -> dict:
    """A line geom's ``label=`` dict with keys that are raw grouping values
    (``{"setosa": ...}``) re-keyed by group id; other keys (group ids) are kept."""
    out: dict = {}
    matched = []
    for gid, value in group_values.items():
        try:
            hit = value in label
        except TypeError:  # unhashable value
            hit = False
        if hit:
            out[gid] = label[value]
            matched.append(value)
    for k, v in label.items():
        if not any(k is m or k == m for m in matched):
            out.setdefault(k, v)
    return out


# --------------------------------------------------------------------------- 2
def _facet(plot) -> Facet:
    facet = getattr(plot, "facet", None) if plot is not None else None
    return facet if facet is not None else FacetNull()


def assign_panels(layers: list[LayerData], plot=None) -> pd.DataFrame:
    """Compute the facet's layout table and add ``PANEL`` to every non-empty layer.

    Without a facet (or ``plot=None``) every non-empty layer gets ``PANEL = 1``.
    Otherwise the plot data and each layer's raw data first go through
    ``facet.facet_data`` (computed facet columns; the result replaces
    ``LayerData.raw``), ``facet.compute_layout`` sees them, and
    ``facet.map_data`` gets each layer's evaluated aesthetics plus the
    facet-variable columns of its raw data (dropped again afterwards); an
    aesthetic column named like a facet variable (``facet_wrap("x")``) is
    renamed while ``map_data`` runs, so only the raw data's column facets. Empty
    layers (``geom_function`` without data) get ``PANEL`` after the stat, which
    runs once per panel for them. Returns the layout table.
    """
    facet = _facet(plot)
    frames = []
    try:
        if plot is not None and isinstance(plot.data, pd.DataFrame):
            frames.append(facet.facet_data(plot.data))
        for ld in layers:
            if isinstance(ld.raw, pd.DataFrame):
                ld.raw = facet.facet_data(ld.raw)
                frames.append(ld.raw)
    except (ValueError, KeyError, TypeError) as e:
        raise BuildError(f"{facet!r}: {e}") from e
    try:
        layout = facet.compute_layout(frames)
    except (ValueError, KeyError, TypeError) as e:
        raise BuildError(f"{facet!r}: {e}") from e
    missing = [c for c in _facet_base.LAYOUT_COLUMNS if c not in layout.columns]
    if missing:
        raise BuildError(f"{facet!r}: layout table lacks column(s) {missing}")
    layout = layout.reset_index(drop=True)
    for ld in layers:
        if not len(ld.data):
            continue
        if facet.is_null:
            ld.data = facet.map_data(ld.data, layout)
            continue
        raw = ld.raw if isinstance(ld.raw, pd.DataFrame) else pd.DataFrame()
        extra = [v for v in facet.vars() if v in raw.columns]
        df = ld.data
        hidden = {v: f"..facet_aes.{v}" for v in facet.vars() if v in df.columns}
        if hidden:  # an aesthetic named like a facet variable: out of map_data's way
            df = df.rename(columns=hidden)
        if extra:
            if len(raw) != len(df):
                raise BuildError(
                    f"{ld.label}: cannot facet: the aesthetics have {len(df)} rows but the "
                    f"data has {len(raw)}"
                )
            df = df.assign(**{v: raw[v].to_numpy() for v in extra})
        df = facet.map_data(df, layout).drop(columns=extra)
        if hidden:
            df = df.rename(columns={tmp: v for v, tmp in hidden.items()})
        ld.data = df.reset_index(drop=True)
    return layout


# --------------------------------------------------------------------------- 3
def _position_scale(plot, axis: str, layers: list[LayerData] | None = None):
    """The user's scale for ``axis`` (cloned), else a default chosen by dtype:
    discrete if any layer maps a non-numeric (non-bool) column to an ``axis``
    aesthetic, continuous otherwise. ``datetime64`` / date-object columns get a
    date or date-time scale (:func:`~ggmanim.scales.date.date_kind`)."""
    cols = [ld.data[c] for ld in layers or () for c in POSITION_AES[axis] if c in ld.data.columns]
    user = plot.get_scale(axis)
    if user is not None:
        scale = user.clone()
        if getattr(scale, "is_date", False):
            if scale.unit == "s" and scale.timezone is None:
                tz = data_timezone(cols)
                if tz is not None:  # display in the data's zone, as ggplot2 does
                    scale.set_timezone(tz)
        elif not getattr(scale, "is_discrete", False) and date_kind(cols) is not None:
            raise BuildError(
                f"the {axis} variable holds dates/date-times but the {axis} scale is a "
                f"continuous {type(scale).__name__} (trans {scale.trans.name!r}); use "
                f"scale_{axis}_date() or scale_{axis}_datetime() instead"
            )
        return scale
    kind = date_kind(cols)
    if kind is not None:  # datetime64 / date objects -> date or datetime scale
        scale = ScaleDatePosition(POSITION_AES[axis], unit="D" if kind == "date" else "s")
        tz = data_timezone(cols) if kind == "datetime" else None
        if tz is not None:
            scale.set_timezone(tz)
        return scale
    for ld in layers or ():
        for c in POSITION_AES[axis]:
            if c in ld.data.columns and _is_discrete_col(ld.data[c]):
                return ScaleDiscretePosition(POSITION_AES[axis])
    return ScaleContinuousPosition(POSITION_AES[axis])


def _is_discrete_col(s: pd.Series) -> bool:
    return is_discrete(s) and not pd.api.types.is_bool_dtype(s) and s.notna().any()


def _check_numeric(ld: LayerData, col: str, scale=None) -> None:
    if getattr(scale, "is_discrete", False) or getattr(scale, "is_date", False):
        return
    s = ld.data[col]
    if _is_discrete_col(s):
        axis = "x" if col in POSITION_AES["x"] else "y"
        raise BuildError(
            f"{ld.label}: {col!r} is mapped to a non-numeric variable but the {axis} "
            f"scale is continuous; use scale_{axis}_discrete() or a numeric column"
        )


def setup_position_scales(plot, layers: list[LayerData]) -> dict[str, Any]:
    """Per-build x/y scales (user scales cloned, else discrete or continuous by
    dtype, see :func:`_position_scale`), trained on the
    raw position columns of every layer; then those columns -> transformed space."""
    scales = {"x": _position_scale(plot, "x", layers), "y": _position_scale(plot, "y", layers)}
    checked: set[tuple[int, str]] = set()
    for ld in layers:
        for axis, cols in ld.layer.geom.position_aes_for_training(ld.data).items():
            for c in cols:
                _check_numeric(ld, c, scales[axis])
                checked.add((ld.index, c))
                scales[axis].train(ld.data[c])
    for ld in layers:
        df = ld.data.copy(deep=False)  # whole columns are replaced, never written into
        for axis, aes_names in POSITION_AES.items():
            for c in aes_names:
                if c in df.columns:
                    if (ld.index, c) not in checked:
                        _check_numeric(ld, c, scales[axis])
                    with np.errstate(all="ignore"):
                        df[c] = _pre_stat_map(scales[axis], df[c])
        ld.data = df
    return scales


def _by_panel(df: pd.DataFrame) -> list[tuple[int, pd.Index]]:
    """``[(PANEL, row index), ...]`` of ``df`` (sorted by panel)."""
    if "PANEL" not in df.columns or not len(df):
        return []
    return [(int(p), idx) for p, idx in df.groupby("PANEL", sort=True).groups.items()]


def setup_panel_scales(plot, layers: list[LayerData], layout: pd.DataFrame) -> dict[int, dict]:
    """Per-panel version of :func:`setup_position_scales`: ``{PANEL: {"x", "y"}}``
    from ``facet.init_scales`` (one shared scale per ``SCALE_X``/``SCALE_Y`` id),
    each trained on its panel's raw position columns; then those columns ->
    transformed space with the panel's scales."""
    facet = _facet(plot)
    proto = {"x": _position_scale(plot, "x", layers), "y": _position_scale(plot, "y", layers)}
    panel_scales = facet.init_scales(layout, proto["x"], proto["y"])
    for ld in layers:
        groups = _by_panel(ld.data)
        for axis, cols in ld.layer.geom.position_aes_for_training(ld.data).items():
            for c in cols:
                _check_numeric(ld, c, proto[axis])
                for p, idx in groups:
                    panel_scales[p][axis].train(ld.data.loc[idx, c])
    for ld in layers:
        df = ld.data.copy(deep=False)
        groups = _by_panel(df)
        for axis, aes_names in POSITION_AES.items():
            for c in aes_names:
                if c in df.columns and groups:
                    _check_numeric(ld, c, proto[axis])
                    parts = []
                    for p, idx in groups:
                        with np.errstate(all="ignore"):
                            mapped = _pre_stat_map(panel_scales[p][axis], df.loc[idx, c])
                        parts.append(pd.Series(np.asarray(mapped), index=idx))
                    df[c] = pd.concat(parts).reindex(df.index)
        ld.data = df
    return panel_scales


def _pre_stat_map(scale, values):
    """Data -> transformed space before the stat. With user ``limits`` the scale's
    ``oob`` applies here (ggplot2: ``xlim()`` censors *before* the stat, so a smooth
    is fitted to in-range data only); trained limits cover all data anyway.

    A discrete scale's own "Removed n value(s) outside the limits" warning is
    silenced here: the dropped rows are reported once per layer (with the layer's
    name) by the stat pre-filter or ``geom.handle_na``."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"Removed \d+ value\(s\) outside the limits", category=UserWarning
        )
        if getattr(scale, "limits", None) is not None:
            return scale.map(values)
        return scale.transform(values)


def _remove_missing_for_stat(ld: LayerData, stat) -> None:
    """Drop rows whose positional ``stat.required_aes`` are non-finite (missing or
    censored by scale limits) before a computing stat, with one warning."""
    pos = set(X_AESTHETICS) | set(Y_AESTHETICS)
    cols = [c for c in getattr(stat, "required_aes", ()) if c in pos and c in ld.data.columns]
    if not cols or not len(ld.data):
        return
    ok = np.ones(len(ld.data), dtype=bool)
    for c in cols:
        ok &= np.isfinite(pd.to_numeric(ld.data[c], errors="coerce").to_numpy(dtype=float))
    n = int((~ok).sum())
    if n:
        if not ld.layer.params.get("na_rm", False):
            warnings.warn(
                f"{ld.label}: removed {n} row{'s' if n != 1 else ''} containing missing "
                f"values or values outside the scale range ({type(stat).__name__})",
                stacklevel=3,
            )
        ld.data = ld.data[ok].reset_index(drop=True)


def _compute_layer_panels(
    ld: LayerData, stat, panel_scales: dict[int, dict], layout: pd.DataFrame | None
) -> pd.DataFrame:
    """``stat.compute_layer`` once per panel with that panel's scales; an empty
    layer is computed for every panel of ``layout`` (e.g. ``geom_function``)."""
    pieces = []
    if len(ld.data):
        for p, idx in _by_panel(ld.data):
            res = stat.compute_layer(ld.data.loc[idx], panel_scales[p], **stat.params)
            pieces.append(res if "PANEL" in res.columns else res.assign(PANEL=p))
    else:
        panels = layout["PANEL"] if layout is not None else sorted(panel_scales)
        for p in (int(v) for v in panels):
            res = stat.compute_layer(ld.data, panel_scales[p], **stat.params)
            if len(res):
                pieces.append(res.assign(PANEL=p))
    if not pieces:
        return stat.compute_layer(ld.data, panel_scales[min(panel_scales)], **stat.params)
    return pd.concat(pieces, ignore_index=True)


def compute_stats(
    layers: list[LayerData],
    scales: dict[str, Any],
    panel_scales: dict[int, dict] | None = None,
    layout: pd.DataFrame | None = None,
) -> None:
    """``stat.compute_layer`` per layer, then evaluate ``after_stat`` aesthetics.

    With ``panel_scales`` (several panels) the stat runs per ``PANEL`` with that
    panel's scales (:func:`_compute_layer_panels`)."""
    for ld in layers:
        stat = ld.layer.stat
        if stat is not None:
            _remove_missing_for_stat(ld, stat)
            try:
                if panel_scales is not None:
                    df = _compute_layer_panels(ld, stat, panel_scales, layout)
                else:
                    df = stat.compute_layer(ld.data, scales, **stat.params)
            except BuildError:
                raise
            except Exception as e:
                raise BuildError(f"{ld.label}: {type(stat).__name__} failed: {e}") from e
        else:
            df = ld.data.copy()
        df = df.reset_index(drop=True)
        after = ld.mapping.after_stat()
        for aes_name, value in after.items():
            try:
                df[aes_name] = value.evaluate(df)
            except KeyError as e:
                raise BuildError(
                    f"{ld.label}: {value!r} for aesthetic {aes_name!r}: variable "
                    f"not computed by the stat (columns: {list(df.columns)})"
                ) from e
        if len(df) and "PANEL" not in df.columns:
            df["PANEL"] = 1
        if "group" not in df.columns:
            df = add_group(df)
        ld.data = df


# --------------------------------------------------------------------------- 4
def _non_position_aesthetics(layers: list[LayerData]) -> list[str]:
    """Mapped non-position aesthetics present in some layer's data: the categorical
    ones first (stable legend order), then the others in mapping order."""
    skip = {"group", "PANEL", *X_AESTHETICS, *Y_AESTHETICS}
    out = list(CATEGORICAL_AESTHETICS)
    for ld in layers:
        for aes_name in ld.mapping:
            if aes_name not in skip and aes_name not in out:
                out.append(aes_name)
    return [a for a in out if any(a in ld.data.columns and len(ld.data) for ld in layers)]


def _mapped_column(values: Any, index: pd.Index) -> pd.Series:
    """Scale output as a column: numeric dtype when every value is a number,
    else ``object`` (colours, shape names ...)."""
    return pd.Series(list(values), index=index, dtype=object).infer_objects()


def map_non_position(plot, layers: list[LayerData]) -> dict[str, Any]:
    """Train the scale of every mapped non-position aesthetic on every layer, then
    map the columns to manim values. The scale is the user's, else
    :func:`default_scale_for` (discrete colour/fill/shape/linetype, continuous colour
    gradients for numeric colour/fill, range scales for size/alpha/linewidth);
    aesthetics without a default scale are used as is. Returns ``{aes: trained scale}``."""
    scales: dict[str, Any] = {}
    clones: dict[int, Any] = {}  # one clone per user scale object (shared aesthetics)
    for aes_name in _non_position_aesthetics(layers):
        present = [ld for ld in layers if aes_name in ld.data.columns and len(ld.data)]
        user = plot.get_scale(aes_name)
        if user is not None:
            scale = clones.get(id(user))
            if scale is None:
                scale = clones[id(user)] = user.clone()
        else:
            values = pd.concat([ld.data[aes_name] for ld in present], ignore_index=True)
            try:
                scale = default_scale_for(aes_name, values)
            except (NotImplementedError, TypeError, ValueError) as e:
                names = ", ".join(ld.label for ld in present)
                raise BuildError(f"{names}: {e}") from e
            if scale is None:
                continue
        for ld in present:
            try:
                scale.train(ld.data[aes_name])
            except TypeError as e:
                raise BuildError(f"{ld.label}: aesthetic {aes_name!r}: {e}") from e
        scales[aes_name] = scale
    for ld in layers:  # one (shallow) copy per layer
        todo = [a for a in scales if a in ld.data.columns and len(ld.data)]
        if not todo:
            continue
        df = ld.data.copy(deep=False)
        for aes_name in todo:
            df[aes_name] = _mapped_column(scales[aes_name].map(df[aes_name]), df.index)
        ld.data = df
    return scales


# --------------------------------------------------------------------------- 5
def setup_geom_data(layers: list[LayerData], scales: dict[str, Any] | None = None) -> None:
    """``geom.setup_data`` per layer (e.g. bar extents for ``geom_col``).

    When the x/y position scale is discrete, the params passed to ``setup_data``
    also carry ``_x_discrete``/``_y_discrete = True`` (e.g. so bars default to
    0.9 of the level spacing)."""
    flags = {
        f"_{axis}_discrete": True
        for axis, s in (scales or {}).items()
        if axis in POSITION_AES and getattr(s, "is_discrete", False)
    }
    for ld in layers:
        params = {**ld.layer.params, **flags} if flags else ld.layer.params
        ld.data = ld.layer.geom.setup_data(ld.data, params)


# --------------------------------------------------------------------------- 6
def apply_positions(
    layers: list[LayerData],
    scales: dict[str, Any] | None = None,
    panel_scales: dict[int, dict] | None = None,
) -> None:
    """Run each layer's position adjustment (after :func:`setup_geom_data`).

    ``pos.setup_params(data)``, ``pos.setup_data(data, params)``, then
    ``pos.compute_layer(data, params, scales)`` (split by ``PANEL``); see
    :mod:`ggmanim.positions.base`. ``position_identity`` returns the data
    unchanged. ``scales`` are the build's position scales (read-only here; they
    are retrained afterwards by :func:`train_position_scales`). With
    ``panel_scales`` each panel's rows are adjusted with that panel's scales.
    ``params["layer_index"]`` is the layer's index in the plot (unless the
    position set it), e.g. so ``position_jitter`` seeds each layer differently."""
    for ld in layers:
        pos = get_position(ld.layer.position)
        try:
            params = pos.setup_params(ld.data)
            params.setdefault("layer_index", ld.index)
            data = pos.setup_data(ld.data, params)
            groups = _by_panel(data) if panel_scales is not None else []
            if groups and panel_scales is not None:
                pieces = [
                    pos.compute_layer(data.loc[idx], params, panel_scales[p]) for p, idx in groups
                ]
                ld.data = pd.concat(pieces, ignore_index=True)
            else:
                ld.data = pos.compute_layer(data, params, scales)
        except BuildError:
            raise
        except Exception as e:
            raise BuildError(f"{ld.label}: {type(pos).__name__} failed: {e}") from e


def train_position_scales(
    layers: list[LayerData],
    scales: dict[str, Any],
    panel_scales: dict[int, dict] | None = None,
) -> None:
    """Reset and retrain x/y on the columns each geom draws (transformed space),
    then apply each scale's ``oob`` policy to those columns. With
    ``panel_scales`` every panel's scales are trained on (and applied to) that
    panel's rows only."""
    if panel_scales is not None:
        _train_panel_scales(layers, panel_scales)
        return
    for s in scales.values():
        s.reset()
    for ld in layers:
        for axis, cols in ld.layer.geom.position_aes_for_training(ld.data).items():
            for c in cols:
                scales[axis].train_transformed(ld.data[c])
    for ld in layers:
        df = ld.data
        changed = False
        for axis, aes_names in POSITION_AES.items():
            for c in aes_names:
                if c in df.columns:
                    if not changed:
                        df, changed = df.copy(deep=False), True
                    df[c] = scales[axis].map_transformed(df[c])
        ld.data = df


def _train_panel_scales(layers: list[LayerData], panel_scales: dict[int, dict]) -> None:
    unique = {id(s): s for sc in panel_scales.values() for s in sc.values()}
    for s in unique.values():
        s.reset()
    for ld in layers:
        groups = _by_panel(ld.data)
        for axis, cols in ld.layer.geom.position_aes_for_training(ld.data).items():
            for c in cols:
                for p, idx in groups:
                    panel_scales[p][axis].train_transformed(ld.data.loc[idx, c])
    for ld in layers:
        groups = _by_panel(ld.data)
        if not groups:
            continue
        df = ld.data.copy(deep=False)
        for axis, aes_names in POSITION_AES.items():
            for c in aes_names:
                if c in df.columns:
                    parts = [
                        pd.Series(
                            np.asarray(panel_scales[p][axis].map_transformed(df.loc[idx, c])),
                            index=idx,
                        )
                        for p, idx in groups
                    ]
                    df[c] = pd.concat(parts).reindex(df.index)
        ld.data = df


# --------------------------------------------------------------------------- 7
def _mapping_label(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, AfterStat):
        return value.expr if isinstance(value.expr, str) else None
    return None


def default_labels(plot, layers: list[LayerData]) -> dict[str, Any]:
    """Titles for every mapped aesthetic: the mapped column names (plot mapping
    first, then layers in order), overridden by ``labs()``."""
    labels: dict[str, Any] = {}
    sources = [plot.mapping] + [ld.mapping for ld in layers]
    for mapping in sources:
        for aes_name, value in (mapping or {}).items():
            lab = _mapping_label(value)
            if lab is not None and aes_name not in labels:
                labels[aes_name] = lab
    labels.update(dict(plot.labels))
    return labels


def build_panel(
    plot, scales: dict[str, Any], theme: Theme, labels: dict[str, Any], has_legend: bool
) -> CartesianPanel:
    """The coord builds the panel (axes, grid, ticks, axis titles)."""
    coord = plot.coord if plot.coord is not None else CoordCartesian()
    return coord.build(scales["x"], scales["y"], theme, labels, has_legend=has_legend)


def _axis_side(scale, default: str) -> str:
    """``"top"``/``"right"`` when a position scale asks for the far side (as
    ``coords`` decides it), else ``default``."""
    far = {"bottom": ("top", "right"), "left": ("right", "top")}[default]
    pos = getattr(scale, "position", None) or default
    return ("top" if default == "bottom" else "right") if pos in far else default


def _shared_title(content: Any, theme: Theme) -> Mobject | None:
    if content is None:
        return None
    if isinstance(content, Mobject):
        return content.copy()
    if str(content) == "":
        return None
    return theme.make_text(content, **theme.text_style("axis_title"))


def _scale_title(scale, labels: dict[str, Any], aes: str):
    name = getattr(scale, "name", None) if scale is not None else None
    return name if name is not None else labels.get(aes)


def _measure_key(scales: dict, visible: tuple[bool, ...], strips) -> tuple:
    """What a pass-1 panel's measurements depend on: its scale objects, which
    axes it draws and the sides (and depth) of its strips."""
    return (
        id(scales["x"]),
        id(scales["y"]),
        tuple(visible),
        tuple((side, len(labels)) for side, labels in strips),
    )


def build_panels(
    plot,
    layout: pd.DataFrame,
    panel_scales: dict[int, dict],
    theme: Theme,
    labels: dict[str, Any],
    has_legend: bool,
) -> PanelGrid:
    """One :class:`CartesianPanel` per layout row, laid out in a grid.

    ``coord.build`` gets each panel's scales and a per-panel theme: its size
    (``panel_width``/``panel_height``; the whole grid fits ``theme.panel_size``
    with ``theme.panel_spacing`` between panels), no axis titles (drawn once for
    the grid) and no tick marks/labels where ``facet.draw_axes`` says so (inner
    panels with shared scales). Panels are built twice: a first build measures
    the tick labels and strips round each plotting area, the second uses the
    sizes that leave room for them (the first build is made once per kind of
    panel, see :func:`_measure_key`). ``facet.draw_panels`` adds the strips and
    places the panels; the shared axis titles go round the grid."""
    facet = _facet(plot)
    coord = plot.coord if plot.coord is not None else CoordCartesian()
    nrow, ncol = int(layout["ROW"].max()), int(layout["COL"].max())
    total = theme.panel_size(has_legend)
    spacing = _facet_base.panel_spacing(theme)
    panel_ids = [int(p) for p in layout["PANEL"]]
    cells = {int(r.PANEL): (int(r.ROW), int(r.COL)) for r in layout.itertuples(index=False)}

    def make(p: int, w: float, h: float, vis: tuple[bool, ...]) -> CartesianPanel:
        changes: dict[str, Any] = {
            "panel_width": w,
            "panel_height": h,
            "axis_title_x": False,
            "axis_title_y": False,
        }
        # vis: (horizontal, vertical[, horizontal secondary, vertical secondary])
        if not (vis[0] or vis[2:3] == (True,)):
            changes.update(axis_text_x=False, axis_ticks_x=False)
        if not (vis[1] or vis[3:4] == (True,)):
            changes.update(axis_text_y=False, axis_ticks_y=False)
        sc = panel_scales[p]
        panel = coord.build(sc["x"], sc["y"], theme.replace(**changes), labels, has_legend=False)
        if len(vis) > 2:  # secondary axes (sec_axis): drop the sides not drawn
            _coords._keep_axes(panel, *vis)
        return panel

    # pass 1: nominal sizes, to learn the orientation and measure decorations
    nominal = (total[0] / ncol, total[1] / nrow)
    first = make(panel_ids[0], *nominal, (True, True, True, True))
    flipped = bool(getattr(first, "flipped", False))
    sc1 = panel_scales[panel_ids[0]]
    h_scale, v_scale = (sc1["y"], sc1["x"]) if flipped else (sc1["x"], sc1["y"])
    sides = (_axis_side(h_scale, "bottom"), _axis_side(v_scale, "left"))
    h_col, v_col = ("SCALE_Y", "SCALE_X") if flipped else ("SCALE_X", "SCALE_Y")
    visible = facet.draw_axes(layout, h_col, v_col, *sides)
    # secondary axes (sec_axis) go on the opposite sides; drawn: (h, v, h sec, v sec)
    sec_sides = (
        "top" if sides[0] == "bottom" else "bottom",
        "right" if sides[1] == "left" else "left",
    )
    sec_visible = facet.draw_axes(layout, h_col, v_col, *sec_sides)
    drawn = {p: tuple(visible[p]) + tuple(sec_visible[p]) for p in visible}
    specs = facet.strips(layout)
    thickness = _facet_base.strip_thickness(theme, specs)
    ext = {}
    params = {}
    # Panels with the same scales, axes and strip sides (strip *text* does not
    # matter: strips span the panel and are ``thickness`` deep) measure the same,
    # so pass 1 builds one panel per such kind; ``first`` is reused for its kind.
    measured: dict[tuple, tuple] = {}
    spare: Any = first  # built at the nominal size with every axis drawn, no strips
    spare_kind = _measure_key(sc1, (True, True, True, True), [])[:3]
    for p in panel_ids:
        key = _measure_key(panel_scales[p], drawn[p], specs.get(p, []))
        if key not in measured:
            if spare is not None and key[:3] == spare_kind:
                panel, spare = spare, None
            else:
                panel = make(p, *nominal, drawn[p])
            strips = _facet_base.add_strips(
                panel, specs.get(p, []), thickness, theme, visible[p], sides
            )
            measured[key] = (
                _facet_layout.extents(VGroup(panel, strips), panel.axes),
                panel.panel_params,
            )
        ext[p], params[p] = measured[key]
    col_w, row_w = facet.panel_weights(layout, params)
    widths, heights = _facet_layout.track_sizes(
        layout,
        ext,
        total,
        spacing,
        col_w,
        row_w,
        strip_ext=_facet_base.strip_extents(specs, thickness),
    )

    # pass 2: final sizes
    panels = {
        p: make(p, widths[cells[p][1] - 1], heights[cells[p][0] - 1], drawn[p]) for p in panel_ids
    }
    grid = facet.draw_panels(panels, layout, theme, widths, heights, visible, sides)

    # axis titles, once for the whole grid (theme "x" = the horizontal axis)
    if flipped:
        h_content = _scale_title(sc1["y"], labels, "y")
        v_content = _scale_title(sc1["x"], labels, "x")
    else:
        h_content = _scale_title(sc1["x"], labels, "x")
        v_content = _scale_title(sc1["y"], labels, "y")
    x_title = _shared_title(h_content, theme) if theme.axis_element("axis_title", "x") else None
    y_title = _shared_title(v_content, theme) if theme.axis_element("axis_title", "y") else None
    grid.add_titles(x_title, y_title, theme, sides)
    h_aes, v_aes = ("y", "x") if flipped else ("x", "y")
    _coords._add_secondary_titles(grid, sc1[h_aes], sc1[v_aes], h_aes, v_aes, labels, theme, sides)
    return grid


# --------------------------------------------------------------------------- 8
def draw_layers(
    layers: list[LayerData], panel: CartesianPanel | PanelGrid, theme: Theme
) -> list[tuple[Any, Mobject]]:
    """``geom.draw_panel`` per layer -> ``[(key, mobject), ...]`` in layer order.

    Rows with missing / oob-censored positions are dropped first
    (``geom.handle_na``: one "removed n rows" warning per layer unless ``na_rm``).

    With a :class:`~ggmanim.facets.PanelGrid` each layer is drawn once per panel
    on that panel's rows; the mobject is a :class:`~ggmanim.facets.PanelLayers`
    (one group per panel, ``.panel(row, col)``, ``.marks`` in panel order).

    A ``label=`` dict of a geom with ``label_by == "group"`` (the line geoms'
    direct labels) may be keyed by raw grouping values; they are re-keyed by
    group id here."""
    out: list[tuple[Any, Mobject]] = []
    for ld in layers:
        geom, params = ld.layer.geom, ld.layer.params
        label = params.get("label")
        if (
            isinstance(label, Mapping)
            and ld.group_values
            and getattr(geom, "label_by", "group") == "group"
        ):
            params = {**params, "label": _labels_by_group(label, ld.group_values)}
        data = geom.handle_na(ld.data, params, ld.label) if hasattr(geom, "handle_na") else ld.data
        if not isinstance(panel, PanelGrid):
            mob = geom.draw_panel(data, panel.panel_params, panel, theme, **params)
            out.append((ld.key, mob))
            continue
        cells = {
            int(r.PANEL): (int(r.ROW), int(r.COL)) for r in panel.layout.itertuples(index=False)
        }
        has_panel = "PANEL" in data.columns
        parts = {}
        for p, pan in panel.panel_ids.items():
            rows = data[data["PANEL"] == p] if has_panel else data.iloc[0:0]
            parts[cells[p]] = geom.draw_panel(rows, pan.panel_params, pan, theme, **params)
        out.append((ld.key, PanelLayers(parts)))
    return out


# --------------------------------------------------------------------------- 9
def make_legends(
    plot, layers: list[LayerData], scales: dict[str, Any], theme: Theme, labels: dict[str, Any]
) -> list:
    """Legend objects for the trained non-position scales (``[]`` if hidden)."""
    if theme.legend_position == "none" or not scales:
        return []
    specs = [(ld.layer.geom, dict(ld.mapping), dict(ld.layer.params)) for ld in layers]
    guide_specs = dict(getattr(plot, "guide_specs", None) or {})
    return list(build_legends(scales, specs, theme, labels, guide_specs) or [])


def _text(theme: Theme, content: Any, size: float, element: str | None = None) -> Mobject | None:
    if content is None:
        return None
    if isinstance(content, Mobject):
        return content.copy()
    if str(content) == "":
        return None
    style = theme.text_style(element) if element is not None else {}
    style["font_size"] = size
    return theme.make_text(str(content), **style)


def _title_edges(
    panel: CartesianPanel | PanelGrid, legend: VGroup, theme: Theme
) -> tuple[float, float]:
    """``(left, right)`` x that titles/captions align to (``plot_title_position``)."""
    if getattr(theme, "plot_title_position", "panel") == "plot":
        whole = VGroup(panel, legend) if len(legend) else panel
        return whole.get_left()[0], whole.get_right()[0]
    return panel.axes.get_left()[0], panel.axes.get_right()[0]


def _plot_background(content: Mobject, theme: Theme) -> Mobject | None:
    """A filled rectangle behind ``content`` padded by ``theme.plot_margin``."""
    if getattr(theme, "plot_background", None) is None:
        return None
    from manim import DL, UR, ManimColor, Polygon

    t, r, b, l_ = theme.plot_margin
    (x0, y0, _), (x1, y1, _) = content.get_corner(DL), content.get_corner(UR)
    x0, x1, y0, y1 = x0 - l_, x1 + r, y0 - b, y1 + t
    corners = [np.array([x, y, 0.0]) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    rect = Polygon(*corners, stroke_width=0)
    rect.set_fill(ManimColor(theme.plot_background), opacity=theme.plot_background_opacity)
    return rect


def assemble(
    panel: CartesianPanel | PanelGrid,
    drawn,
    legends: list,
    theme: Theme,
    labels: dict[str, Any],
    **extra,
):
    """Lay out the panel, layers, legend and titles in a GGPlot, centred in the
    frame (see :func:`fit_to_frame`)."""
    from .mobjects import GGPlot

    legend = VGroup()
    if legends:
        legend = place_legends(panel, legends, theme)
    title = _text(theme, labels.get("title"), theme.title_size, "title")
    subtitle = _text(theme, labels.get("subtitle"), theme.axis_title_size, "subtitle")
    caption = _text(theme, labels.get("caption"), theme.axis_text_size, "caption")
    left, right = _title_edges(panel, legend, theme)
    # titles/caption clear the legend too (ggplot2's title row spans the whole table)
    body = VGroup(panel, legend) if len(legend) else panel
    if subtitle is not None:
        subtitle.next_to(body, UP, buff=0.2)
        subtitle.set_x(left, direction=LEFT)
    if title is not None:
        title.next_to(subtitle if subtitle is not None else body, UP, buff=0.2)
        title.set_x(left, direction=LEFT)
    if caption is not None:
        caption.next_to(body, DOWN, buff=0.2)
        caption.set_x(right, direction=RIGHT)
    plot = GGPlot(
        panel=panel,
        layers=drawn,
        legend=legend,
        legends=legends,
        title=title,
        subtitle=subtitle,
        caption=caption,
        **extra,
    )
    background = _plot_background(plot, theme)
    if background is not None:
        plot.background = background
        plot.add_to_back(background)
        return fit_to_frame(plot, margin=0.0)
    return fit_to_frame(plot, margin=getattr(theme, "plot_margin", 0.2))


def fit_to_frame(plot: Mobject, margin: float | tuple[float, ...] = 0.2) -> Mobject:
    """Centre ``plot`` (its whole bounding box: panel, titles, legend) in the
    frame inside ``margin`` (a number or ``(top, right, bottom, left)``), shrinking
    it only if it would not fit."""
    from manim import config

    if isinstance(margin, (int, float)):
        t = r = b = l_ = float(margin)
    else:
        t, r, b, l_ = (float(m) for m in margin)
    centre = np.array([(l_ - r) / 2, (b - t) / 2, 0.0])
    plot.move_to(centre)
    max_w, max_h = config.frame_width - l_ - r, config.frame_height - t - b
    factor = min(1.0, max_w / max(plot.width, 1e-9), max_h / max(plot.height, 1e-9))
    if factor < 1.0:
        plot.scale(factor, about_point=centre)
    return plot


# --------------------------------------------------------------------------- all
def build_plot(plot, *, theme: Theme | None = None):
    """Run the whole pipeline on ``plot`` and return a :class:`GGPlot`.

    ``theme`` overrides the plot's theme for this build only.
    """
    if theme is None:
        theme = plot.theme if plot.theme is not None else theme_default()

    layers = resolve_layers(plot)
    layout = assign_panels(layers, plot)
    if not _facet(plot).is_null:
        return _build_faceted(plot, layers, layout, theme)
    pos_scales = setup_position_scales(plot, layers)
    compute_stats(layers, pos_scales)
    np_scales = map_non_position(plot, layers)
    setup_geom_data(layers, pos_scales)
    apply_positions(layers, pos_scales)
    train_position_scales(layers, pos_scales)

    labels = default_labels(plot, layers)
    legends = make_legends(plot, layers, np_scales, theme, labels)
    panel = build_panel(plot, pos_scales, theme, labels, has_legend=bool(legends))
    drawn = draw_layers(layers, panel, theme)
    return assemble(
        panel,
        drawn,
        legends,
        theme,
        labels,
        scales={**pos_scales, **np_scales},
        layer_data={ld.key: ld.data for ld in layers},
        layer_geoms={ld.key: ld.layer.geom for ld in layers},
    )


def _build_faceted(plot, layers: list[LayerData], layout: pd.DataFrame, theme: Theme):
    """:func:`build_plot` for a facet with a layout table (any number of panels):
    per-panel position scales, stats and positions; :func:`build_panels`."""
    facet = _facet(plot)
    panel_scales = setup_panel_scales(plot, layers, layout)
    first = panel_scales[min(panel_scales)]
    compute_stats(layers, first, panel_scales, layout)
    np_scales = map_non_position(plot, layers)
    setup_geom_data(layers, first)
    apply_positions(layers, first, panel_scales)
    train_position_scales(layers, first, panel_scales)
    for ld in layers:
        ld.data = facet.finish_data(ld.data, layout)

    labels = default_labels(plot, layers)
    legends = make_legends(plot, layers, np_scales, theme, labels)
    grid = build_panels(plot, layout, panel_scales, theme, labels, has_legend=bool(legends))
    drawn = draw_layers(layers, grid, theme)
    out = assemble(
        grid,
        drawn,
        legends,
        theme,
        labels,
        scales={**first, **np_scales},
        layer_data={ld.key: ld.data for ld in layers},
        layer_geoms={ld.key: ld.layer.geom for ld in layers},
    )
    cells = {int(r.PANEL): (int(r.ROW), int(r.COL)) for r in layout.itertuples(index=False)}
    out._attach_panel_grid(grid, {cells[p]: dict(s) for p, s in panel_scales.items()})
    return out
