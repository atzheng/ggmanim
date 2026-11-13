"""Built-plot mobjects: :class:`GGPlot`, the result of ``ggplot.build()``."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    scales : ``{aesthetic: trained scale}`` used for this build
    layer_data : ``{layer key: DataFrame}`` the data each geom drew (transformed space)

    ``plot["axes"]``, ``plot["layers"]`` and ``plot["legends"]`` give the legacy
    dict-style access; integer/slice indices index the submobjects as usual.
    """

    def __init__(self, panel, layers: Iterable[Tuple[Any, Mobject]], legend: VGroup,
                 legends: Optional[List[Any]] = None, title: Optional[Mobject] = None,
                 subtitle: Optional[Mobject] = None, caption: Optional[Mobject] = None,
                 scales: Optional[Dict[str, Any]] = None,
                 layer_data: Optional[Dict[Any, Any]] = None, **kwargs):
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
        self.add(panel, *self.layers.values(), self.legend)
        for m in (title, subtitle, caption):
            if m is not None:
                self.add(m)

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
                raise KeyError(
                    f"unknown key {key!r}; use one of {sorted(self._KEYS)}"
                ) from None
        return super().__getitem__(key)

    def __repr__(self):
        return f"GGPlot(layers={list(self.layers)!r}, legends={len(self.legends)})"
