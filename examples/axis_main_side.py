import numpy as np
import pandas as pd
from manim import Scene

from ggmanim import aes, ggplot, scale_x_continuous, scale_y_continuous, theme


class AxisMainSide(Scene):
    def construct(self):
        df = pd.DataFrame({"x": np.arange(6.0), "y": np.arange(6.0) ** 2})
        p = (
            ggplot(df, aes("x", "y"))
            .geom_point()
            .add(scale_x_continuous(name="x (top)", position="top"))
            .add(scale_y_continuous(name="y (right)", position="right"))
            .add(theme(axis_text_x_top=True, axis_title_y_right=False))
        )
        self.add(p.build())
