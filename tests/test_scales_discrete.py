import numpy as np
import pandas as pd
import pytest
from manim import ManimColor

from ggmanim.palettes import BRIGHT, MUTED, hue_pal
from ggmanim.scales.discrete import (
    AES_TO_MANIM,
    CATEGORICAL_AESTHETICS,
    NA_COLOR,
    ScaleDiscrete,
    ScaleDiscreteIdentity,
    ScaleDiscreteManual,
    default_scale_for,
    is_discrete,
    scale_color_discrete,
    scale_color_identity,
    scale_color_manual,
    scale_colour_discrete,
    scale_colour_identity,
    scale_colour_manual,
    scale_fill_discrete,
    scale_fill_identity,
    scale_fill_manual,
)

# -- levels ------------------------------------------------------------------


def test_levels_sorted_for_plain_values():
    s = scale_color_discrete()
    s.train(pd.Series(["b", "c", "a", "b", None]))
    assert s.get_limits() == ["a", "b", "c"]


def test_levels_follow_categorical_order():
    s = scale_color_discrete()
    cat = pd.Categorical(["low", "high", "mid"], categories=["low", "mid", "high"], ordered=True)
    s.train(pd.Series(cat))
    assert s.get_limits() == ["low", "mid", "high"]


def test_categorical_drops_unused_by_default():
    cat = pd.Categorical(["z", "x"], categories=["z", "y", "x"])
    s = scale_color_discrete()
    s.train(cat)
    assert s.get_limits() == ["z", "x"]
    s2 = scale_color_discrete(drop=False)
    s2.train(cat)
    assert s2.get_limits() == ["z", "y", "x"]


def test_training_across_layers_unions_levels():
    s = scale_color_discrete()
    s.train(pd.Series(["b", "a"]))
    s.train(np.array(["d", "c", "a"]))
    assert s.get_limits() == ["a", "b", "c", "d"]


def test_numeric_levels_are_python_scalars_and_sorted():
    s = scale_color_discrete()
    s.train(pd.Series([3, 1, 2, 1]))
    assert s.get_limits() == [1, 2, 3]
    assert all(type(x) is int for x in s.get_limits())


def test_mixed_type_levels_do_not_crash():
    s = scale_color_discrete()
    s.train(["b", 1, "a"])
    assert set(s.get_limits()) == {"a", "b", 1}


def test_user_limits_override_training():
    s = scale_color_discrete(limits=["c", "a"])
    s.train(["a", "b", "c"])
    assert s.get_limits() == ["c", "a"]
    out = s.map(["a", "b", "c"])
    assert out[1] == NA_COLOR


# -- mapping -----------------------------------------------------------------


def test_map_returns_manim_colors():
    s = scale_color_discrete(palette=BRIGHT)
    s.train(["a", "b"])
    out = s.map(["b", "a", "b"])
    assert isinstance(out, np.ndarray) and out.dtype == object and out.shape == (3,)
    assert all(type(c) is ManimColor for c in out)
    assert list(out) == [BRIGHT[1], BRIGHT[0], BRIGHT[1]]


def test_map_hex_string_palette_is_coerced_to_manim_color():
    s = scale_fill_discrete(palette=["#FF0000", "#00FF00"])
    s.train(["x", "y"])
    out = s.map(["y"])
    assert type(out[0]) is ManimColor
    assert out[0].to_hex() == "#00FF00"


def test_map_series_preserves_index():
    s = scale_color_discrete()
    s.train(["a", "b"])
    ser = pd.Series(["b", "a"], index=[10, 20])
    out = s.map(ser)
    assert isinstance(out, pd.Series)
    assert list(out.index) == [10, 20]
    assert out[10] == hue_pal()(2)[1]


def test_default_palette_is_hue():
    s = scale_color_discrete()
    s.train(["a", "b", "c"])
    assert [c.to_hex() for c in s.get_palette_values()] == ["#F8766D", "#00BA38", "#619CFF"]


def test_palette_by_name():
    s = scale_color_discrete(palette="muted")
    s.train(["a", "b"])
    assert s.get_palette_values() == MUTED[:2]


def test_missing_and_unknown_values_map_to_na():
    s = scale_color_discrete()
    s.train(["a"])
    out = s.map(["a", None, np.nan, "zzz"])
    assert out[1] == NA_COLOR and out[2] == NA_COLOR and out[3] == NA_COLOR
    s2 = scale_color_discrete(na_value="#000000")
    s2.train(["a"])
    assert s2.map(["q"])[0] == ManimColor("#000000")


def test_too_many_levels_raises():
    s = scale_color_discrete(palette=BRIGHT)
    s.train([f"l{i}" for i in range(len(BRIGHT) + 1)])
    with pytest.raises(ValueError, match="8 values but 9 levels"):
        s.map(["l0"])


def test_too_many_levels_cycles_when_requested():
    s = scale_color_discrete(palette=BRIGHT, cycle=True)
    levels = [f"l{i:02d}" for i in range(len(BRIGHT) + 1)]
    s.train(levels)
    out = s.map(levels)
    assert out[len(BRIGHT)] == BRIGHT[0]


def test_mapping_stable_across_layers_sharing_scale():
    s = scale_color_discrete()
    layer1 = pd.Series(["a", "c"])
    layer2 = pd.Series(["b", "c", "a"])
    s.train(layer1)
    s.train(layer2)
    m1 = s.map(layer1)
    m2 = s.map(layer2)
    assert m1[1] == m2[1]  # "c"
    assert m1[0] == m2[2]  # "a"
    # mapping before vs after an unrelated map call is identical
    assert list(s.map(layer1)) == list(m1)


def test_retraining_invalidates_lookup():
    s = scale_color_discrete(palette=BRIGHT)
    s.train(["b"])
    assert s.map(["b"])[0] == BRIGHT[0]
    s.train(["a"])
    assert s.map(["b"])[0] == BRIGHT[1]


def test_scalar_map():
    s = scale_color_discrete()
    s.train("a")
    assert s.get_limits() == ["a"]
    assert len(s.map("a")) == 1


# -- breaks / labels ---------------------------------------------------------


def test_breaks_and_labels_default():
    s = scale_color_discrete()
    s.train([2, 1])
    assert s.get_breaks() == [1, 2]
    assert s.get_labels() == ["1", "2"]


def test_user_breaks_subset_of_limits():
    s = scale_color_discrete(breaks=["c", "a", "zzz"], labels=["C!", "A!", "Z!"])
    s.train(["a", "b", "c"])
    assert s.get_breaks() == ["c", "a"]
    assert s.get_labels() == ["C!", "A!"]


def test_labels_dict_and_callable():
    s = scale_color_discrete(labels={"a": "Alpha"})
    s.train(["a", "b"])
    assert s.get_labels() == ["Alpha", "b"]
    s2 = scale_color_discrete(labels=str.upper)
    s2.train(["a", "b"])
    assert s2.get_labels() == ["A", "B"]


def test_labels_length_mismatch():
    s = scale_color_discrete(labels=["x"])
    s.train(["a", "b"])
    with pytest.raises(ValueError):
        s.get_labels()


# -- manual / identity -------------------------------------------------------


def test_manual_dict_mapping():
    s = scale_color_manual(values={"b": "#0000FF", "a": "#FF0000"})
    s.train(["a", "b"])
    out = s.map(["a", "b", "a"])
    assert [c.to_hex() for c in out] == ["#FF0000", "#0000FF", "#FF0000"]
    assert all(type(c) is ManimColor for c in out)


def test_manual_dict_missing_level():
    s = scale_color_manual(values={"a": "#FF0000"})
    s.train(["a", "b"])
    with pytest.raises(ValueError, match="no value for level"):
        s.map(["a"])


def test_manual_list_mapping_and_insufficient():
    s = scale_fill_manual(values=["#111111", "#222222"])
    assert s.aesthetics == ("fill",)
    s.train(["y", "x"])
    assert [c.to_hex() for c in s.map(["x", "y"])] == ["#111111", "#222222"]
    s2 = scale_fill_manual(values=["#111111"])
    s2.train(["x", "y"])
    with pytest.raises(ValueError, match="Insufficient values"):
        s2.map(["x"])


def test_identity_scale():
    s = scale_color_identity()
    assert isinstance(s, ScaleDiscreteIdentity)
    assert s.guide == "none"
    s.train(["#FF0000", "#00FF00"])
    out = s.map(pd.Series(["#00FF00", "#FF0000", None]))
    assert out[0].to_hex() == "#00FF00"
    assert type(out[0]) is ManimColor
    assert out[2] == NA_COLOR
    assert scale_fill_identity().aesthetics == ("fill",)


def test_non_colour_scale_values_not_coerced():
    s = ScaleDiscrete("shape", palette=["circle", "square"])
    s.train(["a", "b"])
    assert list(s.map(["b"])) == ["square"]
    assert s.na_value is None


# -- constructors / aliases / plumbing ----------------------------------------


def test_constructors_and_aesthetics():
    assert scale_color_discrete().aesthetics == ("color",)
    assert scale_fill_discrete().aesthetics == ("fill",)
    assert isinstance(scale_color_manual(values=[]), ScaleDiscreteManual)
    assert scale_color_discrete(aesthetics=("colour", "fill")).aesthetics == ("color", "fill")


def test_british_aliases():
    assert scale_colour_discrete.__name__ == "scale_colour_discrete"
    assert scale_colour_discrete().aesthetics == ("color",)
    assert isinstance(scale_colour_manual(values={"a": "red"}), ScaleDiscreteManual)
    assert isinstance(scale_colour_identity(), ScaleDiscreteIdentity)


def test_name_attribute():
    assert scale_color_discrete(name="Group").name == "Group"
    assert scale_color_discrete().name is None


def test_reset_and_clone():
    s = scale_color_discrete(palette=BRIGHT, name="g")
    s.train(["a", "b"])
    c = s.clone()
    assert c is not s and c.get_limits() == [] and c.name == "g" and c.palette is BRIGHT
    assert s.get_limits() == ["a", "b"]  # original untouched
    c.train(["z"])
    assert s.get_limits() == ["a", "b"]
    s.reset()
    assert s.get_limits() == []


def test_add_to_calls_with_scale():
    class FakePlot:
        def _with_scale(self, scale):
            self.got = scale
            return "new-plot"

    p = FakePlot()
    s = scale_color_discrete()
    assert s.add_to(p) == "new-plot"
    assert p.got is s


def test_aes_to_manim():
    assert AES_TO_MANIM == {"color": "color", "fill": "fill_color"}


# -- defaults ----------------------------------------------------------------


def test_categorical_aesthetics_constant():
    assert CATEGORICAL_AESTHETICS == ("color", "fill", "shape", "linetype")


@pytest.mark.parametrize(
    "values,expected",
    [
        (pd.Series(["a", "b"]), True),
        (pd.Series([1.0, 2.0]), False),
        (pd.Series([1, 2]), False),
        (pd.Series([True, False]), True),
        (pd.Series(pd.Categorical([1, 2])), True),
        (np.array(["x"]), True),
        (["a", None], True),
        ([1, 2, None], False),
        ("a", True),
    ],
)
def test_is_discrete(values, expected):
    assert is_discrete(values) is expected


def test_default_scale_for_strings():
    s = default_scale_for("colour", pd.Series(["a", "b"]))
    assert isinstance(s, ScaleDiscrete)
    assert s.aesthetics == ("color",)
    s.train(pd.Series(["a", "b"]))
    assert all(isinstance(c, ManimColor) for c in s.map(["a", "b"]))


def test_default_scale_for_shape_and_linetype():
    s = default_scale_for("shape", ["a", "b"])
    s.train(["a", "b"])
    assert list(s.map(["a", "b"])) == ["circle", "triangle"]
    lt = default_scale_for("linetype", ["a"])
    lt.train(["a"])
    assert list(lt.map(["a"])) == ["solid"]


def test_default_scale_for_numeric_colour_is_continuous():
    from ggmanim.scales.colour import ScaleContinuousColour

    s = default_scale_for("color", pd.Series([1.0, 2.5]))
    assert isinstance(s, ScaleContinuousColour) and s.aesthetics == ("color",)
    f = default_scale_for("fill", np.array([1, 2]))
    assert isinstance(f, ScaleContinuousColour) and f.aesthetics == ("fill",)
    # booleans stay discrete
    assert isinstance(default_scale_for("color", [True, False]), ScaleDiscrete)


def test_default_scale_for_numeric_shape_raises():
    with pytest.raises(TypeError):
        default_scale_for("shape", [1, 2])


def test_default_scale_for_aes_without_default_is_none():
    assert default_scale_for("weight", [1.0, 2.0]) is None
    assert default_scale_for("label", ["a"]) is None


def test_default_scale_for_discrete_size_warns():
    with pytest.warns(UserWarning, match="Using size for a discrete variable"):
        s = default_scale_for("size", ["a"])
    assert isinstance(s, ScaleDiscrete)


def test_scale_base_if_available():
    try:
        from ggmanim.scales.base import Scale
    except ImportError:
        pytest.skip("Scale base (ggm-s11.2) not on this branch")
    assert issubclass(ScaleDiscrete, Scale)
