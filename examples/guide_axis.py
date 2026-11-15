"""Crowded discrete labels: guide_axis(n_dodge=) and guide_axis(check_overlap=).

Render a still::

    manim -ql -s examples/guide_axis.py Dodge
    manim -ql -s examples/guide_axis.py CheckOverlap
"""

import pandas as pd
from manim import Scene

from ggmanim import aes, geom_col, ggplot, guide_axis, scale_x_discrete

DATA = pd.DataFrame(
    {
        "country": [
            "Argentina",
            "Bangladesh",
            "Cambodia",
            "Denmark",
            "Ethiopia",
            "Finland",
            "Guatemala",
            "Honduras",
            "Indonesia",
            "Jamaica",
        ],
        "value": [3, 5, 2, 6, 4, 7, 3, 5, 6, 2],
    }
)


def make_plot(guide):
    return ggplot(DATA, aes("country", "value")).add(geom_col(), scale_x_discrete(guide=guide))


class Dodge(Scene):
    def construct(self):
        self.add(make_plot(guide_axis(n_dodge=2)).build())


class CheckOverlap(Scene):
    def construct(self):
        self.add(make_plot(guide_axis(check_overlap=True)).build())
