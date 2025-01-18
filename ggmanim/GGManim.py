from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from copy import deepcopy

from manim import Axes, VDict, VGroup, Dot
import funcy as f
import pandas as pd
import numpy as np


CATEGORICAL_AES = {
    "color",
    "fill_color",
    "linetype",
    "shape",
}


@dataclass
class Layer(object):
    data: pd.DataFrame
    aes: Dict[str, str]
    geom: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


# TODO support literal columns


MUTED = np.asarray(
    [
        "332288",
        "88CCEE",
        "44AA99",
        "117733",
        "999933",
        "DDCC77",
        "CC6677",
        "882255",
        "AA4499",
    ]
)
BRIGHT = np.asarray(
    ["77AADD", "99DDFF", "44BB99", "BBCC33", "AAAA00", "EEDD88", "EE8866", "FFAABB"],
)


class GGManim(object):
    def __init__(self, data=None, aes=None, **kwargs):
        super(GGManim, self).__init__(**kwargs)
        self.data = deepcopy(data)
        self.aes = aes
        self.layers = {}
        self.axis_kwargs = {}
        self.palette = MUTED

    def preprocess_data(self, data: Optional[pd.DataFrame]):
        data = data or self.data
        if data is None:
            raise ValueError("No data provided")
        return data

    def add_layer(self, geom, data=None, aes=None, layer_id=None, **kwargs):
        self.layers[layer_id or len(self.layers)] = Layer(
            data=data or self.data,
            aes=aes or self.aes,
            geom=geom,
            kwargs=kwargs,
        )
        return self

    def geom_point(self, **kwargs):
        return self.add_layer("geom_point", **kwargs)

    def geom_function(self, **kwargs):
        return self.add_layer("geom_function", **kwargs)

    def geom_line(self, **kwargs):
        self.add_layer("geom_line", **kwargs)

    def build(self, **kwargs):
        """
        Build the plot
        kwargs: additional arguments to pass to Axes
        """
        x_min = min(
            (layer.data[layer.aes["x"]].min() for layer in self.layers.values())
        )
        x_max = max(
            (layer.data[layer.aes["x"]].max() for layer in self.layers.values())
        )
        y_min = min(
            (layer.data[layer.aes["y"]].min() for layer in self.layers.values())
        )
        y_max = max(
            (layer.data[layer.aes["y"]].max() for layer in self.layers.values())
        )
        plot = VDict()
        axes = Axes(
            x_range=[x_min, x_max], y_range=[y_min, y_max], **self.axis_kwargs, **kwargs
        )
        plot.add([("axes", axes)])
        layers = VDict(
            {
                layer_id: getattr(self, f"build_{layer.geom}")(axes, layer)
                for layer_id, layer in self.layers.items()
            }
        )
        plot.add([("layers", layers)])
        return plot

    def color_mapping(self, x: pd.Series):
        unq = x.unique()
        unq_idx = pd.DataFrame(x=unq, idx=range(len(unq)) % len(self.palette)).merge(
            pd.DataFrame(x), on="x"
        )[["idx"]]
        return self.palette[unq_idx]

    def get_aes(self, data: pd.DataFrame, aes_name: str, aes_val: str):
        if aes_name in ["color", "fill_color"]:
            return self.color_mapping(data[aes_val])
        else:
            return data[aes_val]

    def get_all_aes(self, data: pd.DataFrame, aes: Dict):
        return pd.DataFrame.from_dict(
            {k: self.get_aes(data, k, v) for k, v in aes.items()}
        )

    def build_geom_point(self, axes: Axes, layer: Layer):
        aes = layer.aes
        data = layer.data
        aes_vals = self.get_all_aes(data, aes)
        dots = VGroup(
            *[
                Dot(
                    axes.coords_to_point(row["x"], row["y"]),
                    **f.omit(row, ["x", "y"]),
                    **layer.kwargs,
                )
                for row in aes_vals.to_dict("records")
            ]
        )
        return dots

    def build_geom_function(self, axes: Axes, layer: Layer):
        return axes.plot(layer.kwargs["function"], **f.omit(layer.kwargs, ["function"]))
