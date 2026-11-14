"""ggm-g6f: facet variables named like an aesthetic; strip_placement validated by Theme."""

import pandas as pd
import pytest

from ggmanim import aes, facet_grid, facet_wrap, geom_point, ggplot, theme
from ggmanim.theme import Theme


def _df():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 2.0, 1.0], "g": list("aabb")})


def test_facet_wrap_on_a_variable_mapped_to_x():
    plot = ggplot(_df(), aes("x", "y")).add(geom_point(), name="p").add(facet_wrap("x")).build()
    assert len(plot.panels) == 4
    assert [len(plot.layers["p"].panel(*c).marks) for c in sorted(plot.panels)] == [1, 1, 1, 1]
    # the aesthetic column is the layer's own again (no leftover temporary names)
    assert list(plot.layer_data["p"].columns) == ["x", "y", "group", "PANEL"]


def test_facet_on_x_when_x_maps_another_column():
    # x aesthetic = column "y"; facet on the data column "x"
    plot = ggplot(_df(), aes("y", "x")).add(geom_point(), name="p").add(facet_grid(cols="x"))
    built = plot.build()
    assert sorted(built.panels) == [(1, 1), (1, 2), (1, 3), (1, 4)]
    xs = sorted(built.layer_data["p"]["x"])
    assert xs == [1.0, 1.0, 2.0, 2.0]  # the aesthetic, not the facet values


def test_layer_without_the_facet_column_is_repeated_even_with_x_aesthetic():
    other = pd.DataFrame({"u": [2.5], "v": [1.5]})
    plot = (
        ggplot(_df(), aes("x", "y"))
        .add(geom_point())
        .add(geom_point(aes("u", "v"), data=other), name="o")
        .add(facet_wrap("x"))
        .build()
    )
    assert all(len(plot.layers["o"].panel(*c).marks) == 1 for c in plot.panels)


@pytest.mark.parametrize("bad", ["middle", None, 1])
def test_strip_placement_validated_at_theme_creation(bad):
    with pytest.raises(ValueError, match="strip_placement"):
        Theme(strip_placement=bad)
    with pytest.raises(ValueError, match="strip_placement"):
        theme(strip_placement=bad)


def test_strip_placement_valid_values():
    assert Theme(strip_placement="outside").strip_placement == "outside"
    assert Theme().strip_placement == "inside"
