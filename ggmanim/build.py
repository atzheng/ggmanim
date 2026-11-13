"""The build pipeline: plot spec -> :class:`~ggmanim.mobjects.GGPlot`.

Mirrors ggplot2's ``ggplot_build`` + ``ggplot_gtable``, one function per step
(PLAN.md §2; the per-layer contract is spelled out in :mod:`ggmanim.geoms.base`):

1. :func:`resolve_layers` -- layer data/mapping (``inherit_aes``), evaluate the
   aesthetics, ``add_group``.
2. :func:`assign_panels` -- add the ``PANEL`` column (one panel in Phase 1).
3. :func:`setup_position_scales` -- per-build position scales, *pre-stat* training
   on the raw layer data, then x-/y-like columns -> transformed space.
   :func:`compute_stats` -- ``stat.compute_layer`` per layer, then ``after_stat``.
4. :func:`map_non_position` -- train (on every layer) and map colour/fill/shape/
   linetype through discrete scales.
5. :func:`apply_positions` -- identity only in Phase 1.
6. :func:`setup_geom_data` + :func:`train_position_scales` -- ``geom.setup_data``,
   then reset and retrain the position scales on what the geoms will draw
   (intercept-only layers contribute only their intercepts) and apply ``oob``.
7. :func:`build_panel` -- ``coord.build`` -> :class:`~ggmanim.coords.CartesianPanel`.
8. :func:`draw_layers` -- ``geom.draw_panel`` per layer.
9. :func:`assemble` -- legends, titles, the :class:`~ggmanim.mobjects.GGPlot`.

:func:`build_plot` runs them all. It is pure: scales are cloned per build, layer
data is copied, user Mobjects (titles) are copied, and the spec is never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from manim import DOWN, LEFT, RIGHT, UP, Mobject, VGroup

# Importing these modules registers every component (geom_*, scale_*, ...).
from . import coords as _coords  # noqa: F401
from . import geoms as _geoms  # noqa: F401
from . import scales as _scales  # noqa: F401
from . import stats as _stats  # noqa: F401
from . import theme as _theme  # noqa: F401
from .aes import AfterStat, Aes
from .coords import CartesianPanel, CoordCartesian
from .layer import Layer
from .scales.continuous import X_AESTHETICS, Y_AESTHETICS, ScaleContinuousPosition
from .scales.discrete import CATEGORICAL_AESTHETICS, default_scale_for
from .stats import add_group, is_discrete
from .theme import Theme, theme_default

from .guides import build_legends, place_legends  # also registers guides()

__all__ = [
    "BuildError",
    "LayerData",
    "build_plot",
    "resolve_layers",
    "assign_panels",
    "setup_position_scales",
    "compute_stats",
    "map_non_position",
    "apply_positions",
    "setup_geom_data",
    "train_position_scales",
    "default_labels",
    "build_panel",
    "draw_layers",
    "assemble",
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
def resolve_layers(plot) -> List[LayerData]:
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
            raise BuildError(f"{ld.label}: data must be a pandas DataFrame, got "
                             f"{type(raw).__name__}")
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
        ld.data = add_group(df.reset_index(drop=True))
        out.append(ld)
    return out


# --------------------------------------------------------------------------- 2
def assign_panels(layers: List[LayerData]) -> None:
    """Add ``PANEL = 1`` to every non-empty layer (facets slot in here later).

    Empty layers (``geom_function`` without data) get ``PANEL`` after the stat.
    """
    for ld in layers:
        if len(ld.data):
            ld.data = ld.data.assign(PANEL=1)


# --------------------------------------------------------------------------- 3
def _position_scale(plot, axis: str):
    user = plot.get_scale(axis)
    if user is not None:
        if getattr(user, "is_discrete", False):
            raise BuildError(f"discrete {axis} scales are not supported yet")
        return user.clone()
    return ScaleContinuousPosition(POSITION_AES[axis])


def _check_numeric(ld: LayerData, col: str) -> None:
    s = ld.data[col]
    if is_discrete(s) and not pd.api.types.is_bool_dtype(s):
        raise BuildError(
            f"{ld.label}: {col!r} is mapped to a non-numeric variable; discrete "
            "position scales are not supported yet"
        )


def setup_position_scales(plot, layers: List[LayerData]) -> Dict[str, Any]:
    """Per-build x/y scales (user scales cloned, else continuous), trained on the
    raw position columns of every layer; then those columns -> transformed space."""
    scales = {"x": _position_scale(plot, "x"), "y": _position_scale(plot, "y")}
    for ld in layers:
        for axis, cols in ld.layer.geom.position_aes_for_training(ld.data).items():
            for c in cols:
                _check_numeric(ld, c)
                scales[axis].train(ld.data[c])
    for ld in layers:
        df = ld.data.copy()
        for axis, aes_names in POSITION_AES.items():
            for c in aes_names:
                if c in df.columns:
                    _check_numeric(ld, c)
                    with np.errstate(all="ignore"):
                        df[c] = scales[axis].transform(df[c])
        ld.data = df
    return scales


def compute_stats(layers: List[LayerData], scales: Dict[str, Any]) -> None:
    """``stat.compute_layer`` per layer, then evaluate ``after_stat`` aesthetics."""
    for ld in layers:
        stat = ld.layer.stat
        if stat is not None:
            try:
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
def map_non_position(plot, layers: List[LayerData]) -> Dict[str, Any]:
    """Train discrete scales for colour/fill/shape/linetype on every layer, then
    map the columns to manim values. Other non-position aesthetics (size, alpha,
    linewidth) are used as is in Phase 1. Returns ``{aes: trained scale}``."""
    scales: Dict[str, Any] = {}
    clones: Dict[int, Any] = {}  # one clone per user scale object (shared aesthetics)
    for aes_name in CATEGORICAL_AESTHETICS:
        present = [ld for ld in layers if aes_name in ld.data.columns and len(ld.data)]
        if not present:
            continue
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
        for ld in present:
            scale.train(ld.data[aes_name])
        scales[aes_name] = scale
    for aes_name, scale in scales.items():
        for ld in layers:
            if aes_name in ld.data.columns and len(ld.data):
                df = ld.data.copy()
                df[aes_name] = pd.Series(np.asarray(scale.map(df[aes_name]), dtype=object),
                                         index=df.index, dtype=object)
                ld.data = df
    return scales


# --------------------------------------------------------------------------- 5
def apply_positions(layers: List[LayerData]) -> None:
    """Position adjustments; only ``"identity"`` exists in Phase 1."""
    for ld in layers:
        pos = ld.layer.position
        name = pos if isinstance(pos, str) or pos is None else getattr(pos, "name", None)
        if name not in (None, "identity", "position_identity"):
            raise BuildError(f"{ld.label}: position {pos!r} is not supported yet")


# --------------------------------------------------------------------------- 6
def setup_geom_data(layers: List[LayerData]) -> None:
    """``geom.setup_data`` per layer (e.g. bar extents for ``geom_col``)."""
    for ld in layers:
        ld.data = ld.layer.geom.setup_data(ld.data, ld.layer.params)


def train_position_scales(layers: List[LayerData], scales: Dict[str, Any]) -> None:
    """Reset and retrain x/y on the columns each geom draws (transformed space),
    then apply each scale's ``oob`` policy to those columns."""
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
                        df, changed = df.copy(), True
                    df[c] = scales[axis].map_transformed(df[c])
        ld.data = df


# --------------------------------------------------------------------------- 7
def _mapping_label(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, AfterStat):
        return value.expr if isinstance(value.expr, str) else None
    return None


def default_labels(plot, layers: List[LayerData]) -> Dict[str, Any]:
    """Titles for every mapped aesthetic: the mapped column names (plot mapping
    first, then layers in order), overridden by ``labs()``."""
    labels: Dict[str, Any] = {}
    sources = [plot.mapping] + [ld.mapping for ld in layers]
    for mapping in sources:
        for aes_name, value in (mapping or {}).items():
            lab = _mapping_label(value)
            if lab is not None and aes_name not in labels:
                labels[aes_name] = lab
    labels.update(dict(plot.labels))
    return labels


def build_panel(plot, scales: Dict[str, Any], theme: Theme, labels: Dict[str, Any],
                has_legend: bool) -> CartesianPanel:
    """The coord builds the panel (axes, grid, ticks, axis titles)."""
    coord = plot.coord if plot.coord is not None else CoordCartesian()
    return coord.build(scales["x"], scales["y"], theme, labels, has_legend=has_legend)


# --------------------------------------------------------------------------- 8
def draw_layers(layers: List[LayerData], panel: CartesianPanel,
                theme: Theme) -> List[Tuple[Any, Mobject]]:
    """``geom.draw_panel`` per layer -> ``[(key, mobject), ...]`` in layer order."""
    out = []
    for ld in layers:
        mob = ld.layer.geom.draw_panel(ld.data, panel.panel_params, panel, theme,
                                       **ld.layer.params)
        out.append((ld.key, mob))
    return out


# --------------------------------------------------------------------------- 9
def make_legends(plot, layers: List[LayerData], scales: Dict[str, Any], theme: Theme,
                 labels: Dict[str, Any]) -> list:
    """Legend objects for the trained non-position scales (``[]`` if hidden)."""
    if theme.legend_position == "none" or not scales:
        return []
    specs = [(ld.layer.geom, dict(ld.mapping), dict(ld.layer.params)) for ld in layers]
    guide_specs = dict(getattr(plot, "guide_specs", None) or {})
    return list(build_legends(scales, specs, theme, labels, guide_specs) or [])


def _text(theme: Theme, content: Any, size: float) -> Optional[Mobject]:
    if content is None:
        return None
    if isinstance(content, Mobject):
        return content.copy()
    if str(content) == "":
        return None
    return theme.make_text(str(content), font_size=size)


def assemble(panel: CartesianPanel, drawn, legends: list, theme: Theme,
             labels: Dict[str, Any], **extra):
    """Lay out the panel (at the origin), layers, legend and titles in a GGPlot."""
    from .mobjects import GGPlot

    legend = VGroup()
    if legends:
        legend = place_legends(panel, legends, theme)
    title = _text(theme, labels.get("title"), theme.title_size)
    subtitle = _text(theme, labels.get("subtitle"), theme.axis_title_size)
    caption = _text(theme, labels.get("caption"), theme.axis_text_size)
    left = panel.get_left()[0]
    if subtitle is not None:
        subtitle.next_to(panel, UP, buff=0.2)
        subtitle.set_x(left, direction=LEFT)
    if title is not None:
        title.next_to(subtitle if subtitle is not None else panel, UP, buff=0.2)
        title.set_x(left, direction=LEFT)
    if caption is not None:
        caption.next_to(panel, DOWN, buff=0.2)
        caption.set_x(panel.get_right()[0], direction=RIGHT)
    return GGPlot(panel=panel, layers=drawn, legend=legend, legends=legends, title=title,
                  subtitle=subtitle, caption=caption, **extra)


# --------------------------------------------------------------------------- all
def build_plot(plot, *, theme: Optional[Theme] = None):
    """Run the whole pipeline on ``plot`` and return a :class:`GGPlot`.

    ``theme`` overrides the plot's theme for this build only.
    """
    if theme is None:
        theme = plot.theme if plot.theme is not None else theme_default()

    layers = resolve_layers(plot)
    assign_panels(layers)
    pos_scales = setup_position_scales(plot, layers)
    compute_stats(layers, pos_scales)
    np_scales = map_non_position(plot, layers)
    apply_positions(layers)
    setup_geom_data(layers)
    train_position_scales(layers, pos_scales)

    labels = default_labels(plot, layers)
    legends = make_legends(plot, layers, np_scales, theme, labels)
    panel = build_panel(plot, pos_scales, theme, labels, has_legend=bool(legends))
    drawn = draw_layers(layers, panel, theme)
    return assemble(panel, drawn, legends, theme, labels,
                    scales={**pos_scales, **np_scales},
                    layer_data={ld.key: ld.data for ld in layers})
