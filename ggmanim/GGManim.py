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
    Rectangle,
    LogBase,
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


def auto_breaks(xmin, xmax):
    break_size = (xmax - xmin) / 5
    oom = 10 ** np.round(np.log10(break_size))
    return oom


@dataclass
class Layer(object):
    data: pd.DataFrame
    aes: Dict[str, str]
    geom: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


class GGManim(object):
    def __init__(self, data=None, aes=None, palette=BRIGHT, **kwargs):
        super(GGManim, self).__init__(**kwargs)
        self.data = data
        self.aes = aes or {}
        self.layers = {}
        self.axis_kwargs = {}
        self.palette = palette
        self.x_label = Tex("") if aes is None else Tex(aes.get("x", ""))
        self.y_label = Tex("") if aes is None else Tex(aes.get("y", ""))
        self.x_range = None
        self.y_range = None
        self.x_log_scale = False
        self.y_log_scale = False

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

    def geom_vline(self, **kwargs):
        return self.add_layer("geom_vline", **kwargs)

    def geom_hline(self, **kwargs):
        return self.add_layer("geom_hline", **kwargs)

    def geom_smooth(self, **kwargs):
        return self.add_layer("geom_smooth", **kwargs)

    def geom_bar(self, **kwargs):
        return self.add_layer("geom_bar", **kwargs)

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
        if len(x_range) == 2:
            self.x_range = [*x_range, auto_breaks(*x_range)]
        elif len(x_range) == 3:
            self.x_range = x_range
        else:
            raise ValueError("x_range must be length 2 or 3")
        return self

    def ylim(self, y_range):
        if len(y_range) == 2:
            self.y_range = [*y_range, auto_breaks(*y_range)]
        elif len(y_range) == 3:
            self.y_range = y_range
        else:
            raise ValueError("y_range must be length 2 or 3")
        return self

    def scale_x_log10(self):
        self.x_log_scale = True
        return self

    def scale_y_log10(self):
        self.y_log_scale = True
        return self

    def build(self, **kwargs):
        """
        Build the plot
        kwargs: additional arguments to pass to Axes
        """

        # build scales
        # -------------------------------------------------------------------------
        if self.x_range is None:
            x_min = min(
                (
                    layer.data[layer.aes["x"]].min()
                    for layer in self.layers.values()
                )
            )
            x_max = max(
                (
                    layer.data[layer.aes["x"]].max()
                    for layer in self.layers.values()
                )
            )
            x_breaks = auto_breaks(x_min, x_max)
            x_range = [x_min, x_max, x_breaks]
            print(x_range)
        else:
            x_range = self.x_range

        if self.y_range is None:
            y_min = min(
                (
                    layer.data[layer.aes["y"]].min()
                    for layer in self.layers.values()
                )
            )
            y_max = max(
                (
                    layer.data[layer.aes["y"]].max()
                    for layer in self.layers.values()
                )
            )
            y_breaks = auto_breaks(y_min, y_max)
            y_range = [y_min, y_max, y_breaks]
        else:
            y_range = self.y_range

        plot = VDict()

        # Configure log scales if enabled
        axis_config = {}
        if self.x_log_scale:
            axis_config["x_axis_config"] = {
                "scaling": LogBase(custom_labels=True)
            }
        if self.y_log_scale:
            axis_config["y_axis_config"] = {
                "scaling": LogBase(custom_labels=True)
            }

        axes = Axes(
            x_range=x_range,
            y_range=y_range,
            **self.axis_kwargs,
            **axis_config,
            **kwargs,
        ).set_color(BLACK)

        scales = {}
        legends = VDict()

        if any("color" in layer.aes for layer in self.layers.values()):
            unique_colors = np.unique(
                np.concatenate(
                    [
                        layer.data[layer.aes["color"]].unique()
                        for layer in self.layers.values()
                        if "color" in layer.aes
                    ]
                )
            )

            if len(unique_colors) > len(self.palette):
                raise ValueError(
                    f"Too many unique colors for palette. Got {len(unique_colors)}, expected {len(self.palette)}"
                )

            color_scale = pd.DataFrame(
                {
                    "x": unique_colors,
                    "palette_id": np.arange(len(unique_colors))
                    % len(self.palette),
                }
            ).set_index("x")

            scales["color"] = lambda x: self.palette[
                color_scale.loc[x, "palette_id"].values
            ]

            legends["color"] = VGroup(
                *f.flatten(
                    [
                        [
                            Dot(color=self.palette[row["palette_id"]]),
                            Tex(str(id)).scale(0.8),
                        ]
                        for id, row in color_scale.iterrows()
                    ]
                )
            ).arrange_in_grid(
                rows=len(color_scale), buff=0.1, cell_alignment=LEFT
            )
            legends["color"].add(
                SurroundingRectangle(
                    legends["color"],
                    buff=0.1,
                    stroke_width=1,
                    stroke_color=BLACK,
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
                layer_id: getattr(self, f"build_{layer.geom}")(
                    scales, axes, layer
                )
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

    def get_aes(
        self,
        scales: Dict[str, Callable],
        data: pd.DataFrame,
        aes_name: str,
        aes_val: str,
    ):
        return scales.get(aes_name, f.identity)(data[aes_val])

    def get_all_aes(
        self, scales: Dict[str, Callable], data: pd.DataFrame, aes: Dict
    ):
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
            if "color" in other_kwargs:
                label.set_color(ManimColor(other_kwargs["color"]))
        else:
            label = VGroup()
        return VGroup(curve, label)

    def build_geom_vline(self, scales, axes: Axes, layer: Layer):
        x = layer.kwargs["xintercept"]
        label = layer.kwargs.get("label", None)
        other_kwargs = f.omit(layer.kwargs, ["xintercept", "label"])
        poscurve = axes.get_vertical_line(
            axes.c2p(x, axes.y_range[1]), **other_kwargs
        )
        negcurve = axes.get_vertical_line(
            axes.c2p(x, axes.y_range[0]), **other_kwargs
        )

        if isinstance(label, str):
            label = Tex(label)

        if label is not None:
            label = (
                label.rotate(PI / 2)
                .next_to(poscurve.get_end(), LEFT, buff=0.0)
                .align_to(poscurve, UP)
            )
            if "color" in other_kwargs:
                label.set_color(ManimColor(other_kwargs["color"]))

        return VGroup(poscurve, negcurve, label or VGroup())

    def build_geom_hline(self, scales, axes: Axes, layer: Layer):
        y = layer.kwargs["yintercept"]
        label = layer.kwargs.get("label", None)
        other_kwargs = f.omit(layer.kwargs, ["yintercept", "label"])
        poscurve = axes.get_horizontal_line(
            axes.c2p(axes.x_range[1], y), **other_kwargs
        )
        negcurve = axes.get_horizontal_line(
            axes.c2p(axes.x_range[0], y), **other_kwargs
        )

        if isinstance(label, str):
            label = Tex(label)

        if label is not None:
            label = label.next_to(poscurve.get_end(), UP, buff=0.0).align_to(
                poscurve, RIGHT
            )
            if "color" in other_kwargs:
                label.set_color(ManimColor(other_kwargs["color"]))

        return VGroup(poscurve, negcurve, label or VGroup())

    def build_geom_smooth(self, scales, axes: Axes, layer: Layer):
        model = layer.kwargs.get("model", LinearRegression())
        aes_vals = self.get_all_aes(scales, layer.data, layer.aes)
        grouping_vars = set(["color", "fill_color"]).intersection(
            layer.aes.keys()
        )

        curves = VGroup()
        if len(grouping_vars) > 0:
            for group, data in aes_vals.groupby(list(grouping_vars)):
                model.fit(data[["x"]], data["y"])
                curves.add(
                    axes.plot(
                        lambda x: model.predict(x.reshape(-1, 1)).reshape(-1),
                        use_vectorized=True,
                        **f.omit(layer.kwargs, ["model"]),
                        **dict(zip(grouping_vars, group)),
                    )
                )
        else:
            model.fit(aes_vals[["x"]], aes_vals["y"])
            curves.add(
                axes.plot(
                    lambda x: model.predict(x.reshape(-1, 1)).reshape(-1),
                    use_vectorized=True,
                    **f.omit(layer.kwargs, ["model"]),
                )
            )

        return curves

    def build_geom_line(self, scales, axes: Axes, layer: Layer):
        aes_vals = self.get_all_aes(scales, layer.data, layer.aes)
        grouping_vars = set(["color", "fill_color"]).intersection(
            layer.aes.keys()
        )
        label = layer.kwargs.get("label", None)
        color = layer.kwargs.get("color", None)
        other_kwargs = f.omit(layer.kwargs, ["label", "color"])

        lines = VGroup()
        if len(grouping_vars) > 0:
            for group, data in aes_vals.groupby(list(grouping_vars)):
                sorted_data = data.sort_values("x")

                group_kwargs = dict(zip(grouping_vars, group))
                line = axes.plot_line_graph(
                    sorted_data["x"].values,
                    sorted_data["y"].values,
                    add_vertex_dots=False,
                    **other_kwargs,
                    **f.omit(group_kwargs, ["color"]),
                )
                line.set_color(ManimColor(group_kwargs["color"]))
                lines.add(line)
        else:
            sorted_data = aes_vals.sort_values("x")
            line = axes.plot_line_graph(
                sorted_data["x"].values,
                sorted_data["y"].values,
                add_vertex_dots=False,
                **other_kwargs,
            )
            if color is not None:
                line.set_color(ManimColor(color))
            lines.add(line)

        if label is not None:
            lbl = Tex(label) if isinstance(label, str) else label
            lbl.next_to(axes.c2p(sorted_data["x"].values[-1], sorted_data["y"].values[-1]), RIGHT)
            if color is not None:
                lbl.set_color(ManimColor(color))
            lines.add(lbl)

        return lines

    def build_geom_bar(self, scales, axes: Axes, layer: Layer):
        aes_vals = self.get_all_aes(scales, layer.data, layer.aes)
        fade = 1 - layer.kwargs.get("alpha", 1)
        width = layer.kwargs.get("width", 0.5)

        bars = VGroup()
        for row in aes_vals.to_dict("records"):
            x_val = row["x"]
            y_val = row["y"]

            # Get the bottom and top points
            bottom = axes.coords_to_point(x_val, 0)
            top = axes.coords_to_point(x_val, y_val)

            # Calculate bar dimensions
            bar_height = np.linalg.norm(top - bottom)
            bar_width = (
                width * axes.x_length / (axes.x_range[1] - axes.x_range[0])
            )

            # Create rectangle
            bar = Rectangle(
                width=bar_width,
                height=bar_height,
                **f.omit(row, ["x", "y"]),
                **f.omit(layer.kwargs, ["alpha", "width"]),
            )

            # Position the bar
            bar.move_to((bottom + top) / 2)
            bar.fade(fade)
            bars.add(bar)

        return bars
