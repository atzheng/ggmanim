from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass, field
from sklearn.linear_model import LinearRegression

from manim import (
    Table,
    BLACK,
    Axes,
    VDict,
    VGroup,
    Dot,
    ManimColor,
    UP,
    DOWN,
    Tex,
    RIGHT,
    LEFT,
    PI,
    Mobject,
    SurroundingRectangle,
)
import funcy as f
import pandas as pd
import numpy as np

from .palettes import *


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


class GGManim(object):
    def __init__(self, data=None, aes=None, **kwargs):
        super(GGManim, self).__init__(**kwargs)
        self.data = data
        self.aes = aes or {}
        self.layers = {}
        self.axis_kwargs = {}
        self.palette = BRIGHT
        self.x_label = Tex("") if aes is None else Tex(aes.get("x", ""))
        self.y_label = Tex("") if aes is None else Tex(aes.get("y", ""))
        self.x_range = None
        self.y_range = None

    def add_layer(self, geom, data=None, aes=None, layer_id=None, **kwargs):
        self.layers[layer_id or len(self.layers)] = Layer(
            data=data or self.data,
            aes=f.select_values(f.notnone, f.merge(self.aes, aes or {})),
            geom=geom,
            kwargs=kwargs,
        )
        return self

    def geom_point(self, **kwargs):
        return self.add_layer("geom_point", **kwargs)

    def geom_function(self, **kwargs):
        return self.add_layer("geom_function", **kwargs)

    def geom_line(self, **kwargs):
        return self.add_layer("geom_line", **kwargs)

    def geom_smooth(self, **kwargs):
        return self.add_layer("geom_smooth", **kwargs)

    def xlab(self, x_label):
        if isinstance(x_label, Mobject):
            self.x_label = x_label
        else:
            self.x_label = Tex(x_label)
        return self

    def ylab(self, y_label):
        if isinstance(y_label, Mobject):
            self.y_label = y_label
        else:
            self.y_label = Tex(y_label)
        return self

    def xlim(self, x_range):
        self.x_range = x_range
        return self

    def ylim(self, y_range):
        self.y_range = y_range
        return self

    def build(self, **kwargs):
        """
        Build the plot
        kwargs: additional arguments to pass to Axes
        """

        # build scales
        # -------------------------------------------------------------------------
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
            x_range=self.x_range or [x_min, x_max],
            y_range=self.y_range or [y_min, y_max],
            **self.axis_kwargs,
            **kwargs,
        )

        scales = {}
        legends = VDict()

        if any("color" in layer.aes for layer in self.layers.values()):
            unique_colors = np.unique(np.concatenate(
                [
                    layer.data[layer.aes["color"]].unique()
                    for layer in self.layers.values()
                    if "color" in layer.aes
                ]
            ))

            if len(unique_colors) > len(self.palette):
                raise ValueError(
                    f"Too many unique colors for palette. Got {len(unique_colors)}, expected {len(self.palette)}"
                )

            color_scale = pd.DataFrame(
                {"x": unique_colors, "palette_id": np.arange(len(unique_colors)) % len(self.palette)}
            ).set_index("x")

            scales["color"] = lambda x: self.palette[color_scale.loc[x, "palette_id"].values]

            legends["color"] = VGroup(
                *f.flatten([
                    [Dot(color=self.palette[row["palette_id"]]), Tex(str(id)).scale(0.8)]
                    for id, row in color_scale.iterrows()
                ])
            ).arrange_in_grid(rows=len(color_scale), buff=0.1, cell_alignment=LEFT)
            legends["color"].add(
                SurroundingRectangle(
                    legends["color"],
                    buff=0.1,
                    stroke_width=1,
                    stroke_color=BLACK
                )
            )


        # build axes
        # -------------------------------------------------------------------------
        axes.add(axes.get_x_axis_label(self.x_label, direction=DOWN, edge=DOWN))
        axes.add(
            axes.get_y_axis_label(
                self.y_label.rotate(PI / 2), direction=LEFT, edge=LEFT
            )
        )
        plot.add([("axes", axes)])

        # build layers
        # -------------------------------------------------------------------------
        layers = VDict(
            {
                layer_id: getattr(self, f"build_{layer.geom}")(scales, axes, layer)
                for layer_id, layer in self.layers.items()
            }
        )
        plot.add([("layers", layers)])

        # build legends
        # -------------------------------------------------------------------------
        legends.arrange(DOWN).next_to(axes, RIGHT)
        plot.add([("legends", legends)])
        return plot

    def color_mapping(self, x: pd.Series):
        unq = x.unique()
        unq_idx = (
            pd.DataFrame({"x": x})
            .merge(
                pd.DataFrame(
                    {"x": unq, "idx": np.arange(len(unq)) % len(self.palette)}
                ),
                on="x",
            )["idx"]
            .values
        )
        return self.palette[unq_idx]

    def get_aes(self, scales: Dict[str, Callable], data: pd.DataFrame, aes_name: str, aes_val: str):
        return scales.get(aes_name, f.identity)(data[aes_val])

    def get_all_aes(self, scales: Dict[str, Callable], data: pd.DataFrame, aes: Dict):
        return pd.DataFrame.from_dict(
            {k: self.get_aes(scales, data, k, v) for k, v in aes.items()}
        )

    def build_geom_point(self, scales, axes: Axes, layer: Layer):
        aes = layer.aes
        data = layer.data
        aes_vals = self.get_all_aes(scales, data, aes)
        fade = 1 - layer.kwargs.get("alpha", 1)

        dots = VGroup(
            *[
                Dot(
                    axes.coords_to_point(row["x"], row["y"]),
                    **f.omit(row, ["x", "y"]),
                    **f.omit(layer.kwargs, ["alpha"]),
                ).fade(fade)
                for row in aes_vals.to_dict("records")
            ]
        )

        return dots

    def build_geom_function(self, scales, axes: Axes, layer: Layer):
        fn = layer.kwargs["function"]
        label = layer.kwargs.get("label", None)
        other_kwargs = f.omit(layer.kwargs, ["function", "label"])
        curve = axes.plot(fn, **other_kwargs)
        if label is not None:
            label = Tex(label).next_to(curve.get_end(), RIGHT)
        else:
            label = VGroup()
        return VGroup(curve, label)

    def build_geom_smooth(self, scales, axes: Axes, layer: Layer):
        model = layer.kwargs.get("model", LinearRegression())
        aes_vals = self.get_all_aes(scales, layer.data, layer.aes)
        grouping_vars = set(["color", "fill_color"]).intersection(layer.aes.keys())

        curves = VGroup()
        if len(grouping_vars) > 0:
            for group, data in aes_vals.groupby(list(grouping_vars)):
                print(group)
                model.fit(data[["x"]], data["y"])
                curves.add(axes.plot(
                    lambda x: model.predict(x.reshape(-1, 1)).reshape(-1),
                    use_vectorized=True,
                    **f.omit(layer.kwargs, ["model"]),
                    **dict(zip(grouping_vars, group))
                ))
        else:
            model.fit(aes_vals[["x"]], aes_vals["y"])
            curves.add(axes.plot(
                lambda x: model.predict(x.reshape(-1, 1)).reshape(-1),
                use_vectorized=True,
                **f.omit(layer.kwargs, ["model"]),
            ))

        return curves
