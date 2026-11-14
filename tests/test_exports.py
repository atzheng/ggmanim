"""The generated export blocks (scripts/gen_exports.py) are up to date and complete."""

import importlib.util
import io
import shutil
import sys
from pathlib import Path

import pytest

import ggmanim
from ggmanim.registry import COMPONENTS

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "gen_exports.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_exports", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("gen_exports", mod)
    spec.loader.exec_module(mod)
    return mod


gen = _load_generator()


def test_export_blocks_up_to_date():
    out = io.StringIO()
    code = gen.run(check=True, out=out)
    assert code == 0, (
        "export blocks are stale; run `python scripts/gen_exports.py`\n" + out.getvalue()
    )


def test_every_component_is_exported():
    missing = sorted(set(COMPONENTS) - set(ggmanim.__all__))
    assert not missing, f"registered components missing from ggmanim.__all__: {missing}"


def test_all_names_resolve():
    for name in ggmanim.__all__:
        assert hasattr(ggmanim, name), name
    assert len(set(ggmanim.__all__)) == len(ggmanim.__all__)
    for pkg in gen.PACKAGES:
        mod = importlib.import_module(f"ggmanim.{pkg}")
        for name in mod.__all__:
            assert hasattr(mod, name), f"ggmanim.{pkg}.{name}"


def test_subpackage_exports_every_public_module():
    for pkg in gen.PACKAGES:
        mod = importlib.import_module(f"ggmanim.{pkg}")
        for path in gen.public_modules(ROOT / "ggmanim" / pkg):
            sub = importlib.import_module(f"ggmanim.{pkg}.{path.stem}")
            assert set(sub.__all__) <= set(mod.__all__), (pkg, path.stem)


def test_top_level_api_kept():
    # names that were exported before the generator existed must stay exported
    for name in ("ggplot", "GGManim", "aes", "after_stat", "labs", "xlab", "ylab", "guides"):
        assert name in ggmanim.__all__
    for name in ("BRIGHT", "Palette", "hue_pal", "viridis_pal", "label_math", "breaks_log"):
        assert name in ggmanim.__all__
    assert "position_identity" in ggmanim.__all__
    # helpers stay in their subpackage
    for name in ("Stat", "GeomPoint", "Scale", "add_group", "scale_limits", "Position"):
        assert name not in ggmanim.__all__


def test_stats_registry_derived_from_classes():
    from ggmanim.stats import STATS, StatFunction, StatIdentity, StatSmooth, _stat_name

    assert STATS == {"function": StatFunction, "identity": StatIdentity, "smooth": StatSmooth}
    assert _stat_name(type("StatSummaryBin", (), {})) == "summary_bin"
    assert _stat_name(type("StatBin2d", (), {})) == "bin_2d"
    assert _stat_name(type("StatQq", (), {"name": "qq_custom"})) == "qq_custom"


# -- the generator itself ---------------------------------------------------------


def _copy_tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(
        ROOT / "ggmanim", root / "ggmanim", ignore=shutil.ignore_patterns("__pycache__")
    )
    return root


def _blocks(root: Path) -> dict:
    return {p.relative_to(root): p.read_text() for p in (root / "ggmanim").rglob("__init__.py")}


def test_generator_idempotent_and_deterministic(tmp_path):
    root = _copy_tree(tmp_path)
    before = _blocks(root)
    assert gen.run(root=root, out=io.StringIO()) == 0
    assert _blocks(root) == before  # already up to date: nothing rewritten
    assert gen.expected_blocks(root) == gen.expected_blocks(root)


def test_new_module_only_needs_the_generator(tmp_path):
    root = _copy_tree(tmp_path)
    (root / "ggmanim" / "geoms" / "zzdummy.py").write_text(
        '__all__ = ["GeomZzGenTest", "geom_zz_gen_test", "ZZ_GEN_TEST_CONST"]\n'
        "GeomZzGenTest = geom_zz_gen_test = ZZ_GEN_TEST_CONST = None\n"
    )
    out = io.StringIO()
    assert gen.run(check=True, root=root, out=out) == 1
    assert "geom_zz_gen_test" in out.getvalue()
    assert gen.run(root=root, out=io.StringIO()) == 0
    geoms_init = (root / "ggmanim" / "geoms" / "__init__.py").read_text()
    top_init = (root / "ggmanim" / "__init__.py").read_text()
    assert "from .zzdummy import ZZ_GEN_TEST_CONST, GeomZzGenTest, geom_zz_gen_test" in geoms_init
    assert '"GeomZzGenTest",' in geoms_init and '"geom_zz_gen_test",' in top_init
    assert '"GeomZzGenTest",' not in top_init  # only the geom_ prefix reaches the top level
    assert gen.run(check=True, root=root, out=io.StringIO()) == 0


def test_generator_repairs_conflicted_block(tmp_path):
    root = _copy_tree(tmp_path)
    init = root / "ggmanim" / "geoms" / "__init__.py"
    good = init.read_text()
    broken = good.replace(
        "from .col import", "<<<<<<< ours\nfrom .area import GeomArea\n=======\nfrom .col import"
    ).replace("__all__ = [", ">>>>>>> theirs\n__all__ = [")
    init.write_text(broken)
    assert gen.run(root=root, out=io.StringIO()) == 0
    assert init.read_text() == good


def test_generator_rejects_bad_modules(tmp_path):
    root = _copy_tree(tmp_path)
    (root / "ggmanim" / "geoms" / "noall.py").write_text("x = 1\n")
    out = io.StringIO()
    assert gen.run(root=root, out=out) == 2
    assert "without __all__" in out.getvalue()
    (root / "ggmanim" / "geoms" / "noall.py").write_text('__all__ = ["GeomPoint"]\n')
    out = io.StringIO()
    assert gen.run(root=root, out=out) == 2
    assert "exported by both" in out.getvalue()


@pytest.mark.parametrize(
    "names, expected",
    [
        (["b", "A", "CONST", "a2", "a10"], ["CONST", "A", "a2", "a10", "b"]),
        (["scale_x_log10", "scale_x_continuous"], ["scale_x_continuous", "scale_x_log10"]),
    ],
)
def test_isort_order(names, expected):
    assert sorted(names, key=gen.isort_key) == expected
