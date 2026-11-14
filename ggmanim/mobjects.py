"""Built-plot mobjects: :class:`GGPlot`, the result of ``ggplot.build()``."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from manim import Mobject, VGroup

__all__ = ["GGPlot", "LayerDict"]


class LayerDict(dict):
    """``{layer key: mobject}`` in layer order. Keys are layer names, or the layer
    index for unnamed layers; an ``int`` that is not a key is also accepted as a
    position (``layers[0]`` is the first layer even when it is named)."""

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        if isinstance(key, int) and not isinstance(key, bool):
            values = list(self.values())
            try:
                return values[key]
            except IndexError:
                pass
        raise KeyError(key)


class GGPlot(VGroup):
    """A built plot.

    Submobjects (back to front): ``panel`` (grid, axes, ticks, tick labels, axis
    titles), one mobject per layer, ``legend``, then the title/subtitle/caption
    when present.

    Attributes
    ----------
    panel : :class:`~ggmanim.coords.CartesianPanel`
    axes : the panel's manim ``Axes``
    layers : :class:`LayerDict` ``{name or index: mobject}``
    legend : ``VGroup`` of placed legends (empty when there is none)
    legends : the guide objects returned by the guides module
    title, subtitle, caption : ``Mobject`` or ``None``
    background : the ``theme.plot_background`` rectangle (first submobject) or ``None``
    scales : ``{aesthetic: trained scale}`` used for this build
    layer_data : ``{layer key: DataFrame}`` the data each geom drew (transformed space)
    layer_geoms : ``{layer key: Geom}`` the geom that drew each layer (picks its animation)

    ``plot["axes"]``, ``plot["layers"]`` and ``plot["legends"]`` give the legacy
    dict-style access; integer/slice indices index the submobjects as usual.
    """

    # region: structure
    def __init__(
        self,
        panel,
        layers: Iterable[tuple[Any, Mobject]],
        legend: VGroup,
        legends: list[Any] | None = None,
        title: Mobject | None = None,
        subtitle: Mobject | None = None,
        caption: Mobject | None = None,
        scales: dict[str, Any] | None = None,
        layer_data: dict[Any, Any] | None = None,
        layer_geoms: dict[Any, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.panel = panel
        self.axes = panel.axes
        self.panel_params = panel.panel_params
        self.layers = LayerDict(layers)
        self.legend = legend if legend is not None else VGroup()
        self.legends = list(legends or [])
        self.title = title
        self.subtitle = subtitle
        self.caption = caption
        self.scales = dict(scales or {})
        self.layer_data = dict(layer_data or {})
        self.layer_geoms = dict(layer_geoms or {})
        self.background: Mobject | None = None  # theme.plot_background (set by build)
        self.add(panel, *self.layers.values(), self.legend)  # type: ignore[arg-type]
        for m in (title, subtitle, caption):
            if m is not None:
                self.add(m)  # type: ignore[arg-type]

    _KEYS = {
        "axes": "axes",
        "panel": "panel",
        "layers": "layers",
        "legends": "legend",
        "legend": "legend",
        "title": "title",
    }

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return getattr(self, self._KEYS[key])
            except KeyError:
                raise KeyError(f"unknown key {key!r}; use one of {sorted(self._KEYS)}") from None
        return super().__getitem__(key)

    def __repr__(self):
        return f"GGPlot(layers={list(self.layers)!r}, legends={len(self.legends)})"

    # endregion

    # region: facets
    # Faceted plots (ggmanim.facets): the build passes a PanelGrid as ``panel``
    # and then calls _attach_panel_grid, so ``panel``/``axes``/``panel_params``
    # are the first panel's while the grid stays the submobject.
    _panel_grid: Any = None
    panel_scales: dict = {}

    @property
    def panels(self) -> dict:
        """``{(row, col): CartesianPanel}``; ``{(1, 1): panel}`` without facets."""
        if self._panel_grid is None:
            return {(1, 1): self.panel}
        return dict(self._panel_grid.panels)

    @property
    def strips(self) -> VGroup:
        """``VGroup`` of every facet strip (empty without facets)."""
        if self._panel_grid is None:
            return VGroup()
        return self._panel_grid.strips

    @property
    def panel_grid(self):
        """The :class:`~ggmanim.facets.PanelGrid` of a faceted plot, else ``None``."""
        return self._panel_grid

    def _attach_panel_grid(self, grid, panel_scales: dict | None = None) -> None:
        first = next(iter(grid.panel_ids.values()))
        self._panel_grid = grid
        self.panel = first
        self.axes = first.axes
        self.panel_params = first.panel_params
        self.panel_scales = dict(panel_scales or {})

    # endregion

    # region: animation
    # See ggmanim.animate. ``layer_geoms`` ({layer key: Geom}) picks each layer's
    # animation; ggmanim.animate.attach_geoms fills it from the spec.
    layer_geoms: dict = {}

    def create(
        self,
        lag_ratio: float = 1.0,
        run_time: float | None = None,
        order: tuple[str, ...] = ("panel", "axes", "layers", "legend", "titles"),
        **kwargs,
    ):
        """One animation revealing the plot stage by stage (panel, axes, each
        layer with its geom's animation, legend, titles); see
        :func:`ggmanim.animate.create_plot` for ``panel_lag``, ``mark_lag`` and
        ``animations``."""
        from .animate import create_plot

        return create_plot(self, lag_ratio=lag_ratio, run_time=run_time, order=order, **kwargs)

    def reveal_layer(self, name, **kwargs):
        """Animation drawing layer ``name`` (:func:`ggmanim.animate.reveal_layer`)."""
        from .animate import reveal_layer

        return reveal_layer(self, name, **kwargs)

    def hide_layer(self, name, **kwargs):
        """Animation removing layer ``name`` (:func:`ggmanim.animate.hide_layer`)."""
        from .animate import hide_layer

        return hide_layer(self, name, **kwargs)

    def mark(self, layer, key) -> Mobject:
        """The mobject drawn for the data row whose ``key`` aesthetic (else row
        index / group id) is ``key`` in ``layer`` (:func:`ggmanim.animate.mark`)."""
        from .animate import mark

        return mark(self, layer, key)

    # endregion

    # region: live
    # See ggmanim.live. ``spec``/``live_theme`` are recorded by
    # ggmanim.live.build_live (and LivePlot / always_rebuild).
    spec: Any = None
    live_theme: Any = None

    def update_data(self, data, layers=None, *, spec=None):
        """Move the marks to the rows of ``data`` (new plot data) in place,
        without rebuilding text or legends; needs fixed x/y limits and identity
        stats (:func:`ggmanim.live.update_data`). Returns ``self``, so it can be
        used in ``add_updater``."""
        from .live import update_data

        return update_data(self, data, layers, spec=spec)

    # endregion
