from manim import ManimColor
import funcy as f
import numpy as np


MUTED = np.asarray(
    f.lmap(
        ManimColor.from_hex,
        [
            "#332288",
            "#117733",
            "#CC6677",
            "#88CCEE",
            "#999933",
            "#882255",
            "#44AA99",
            "#DDCC77",
            "#AA4499",
        ],
    )
)

BRIGHT = np.asarray(
    f.lmap(
        ManimColor.from_hex,
        [
            "#77AADD",
            "#BBCC33",
            "#EE8866",
            "#99DDFF",
            "#AAAA00",
            "#FFAABB",
            "#44BB99",
            "#EEDD88",
        ],
    )
)
