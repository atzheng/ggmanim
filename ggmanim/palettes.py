from manim import ManimColor
import funcy as f
import numpy as np


class HashableManimColor(ManimColor):
    def __hash__(self):
            return hash(self.to_integer())

    def __lt__(self, other):
        return self.to_integer() < other.to_integer()

    def __eq__(self, other):
            return self.to_integer() == other.to_integer()

    def __repr__(self):
            return f"HashableManimColor({self.to_hex()})"

    def __str__(self):
            return self.to_integer()

    def __getitem__(self, key):
            return self.to_integer()[key]

    def __iter__(self):
            return iter(self.to_integer())

    def __len__(self):
            return len(self.to_integer())

    @classmethod
    def from_hex(cls, hex_str):
            return cls(hex_str)


MUTED = np.asarray(
    f.lmap(
        HashableManimColor.from_hex,
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
        HashableManimColor.from_hex,
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
