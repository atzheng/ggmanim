#!/usr/bin/env python
"""Regenerate the export blocks of ggmanim's ``__init__.py`` files.

Usage::

    python scripts/gen_exports.py            # rewrite stale blocks in place
    python scripts/gen_exports.py --check    # exit 1 (and show a diff) if any is stale

Every package listed in :data:`PACKAGES` (``ggmanim/geoms``, ``stats``, ``scales``,
``positions``, ``facets``) and the top-level ``ggmanim/__init__.py`` contain a block
delimited by::

    # -- generated exports (scripts/gen_exports.py) --
    ...
    # -- end generated --

The block holds explicit ``from .x import a, b`` lines plus ``__all__``; never edit it
by hand. On a merge conflict inside a block, just re-run this script (it replaces
everything from the first start marker to the last end marker, conflict markers
included). Anything outside the block is hand-written and left alone.

Rules
-----
* **Package blocks.** Every *public* module of a package (``<pkg>/<name>.py`` whose
  name does not start with ``_``) must define a literal ``__all__``. The block
  imports each module's ``__all__`` (modules in name order) and sets the package's
  ``__all__`` to all of them, sorted. A name exported by two modules of one package
  is an error. Code after the block may extend it (``__all__ += ["STATS"]``).
* **Top-level block.** For each source in :data:`TOP_LEVEL`, the names of its
  ``__all__`` (for a package: of all its public modules) that match the source's
  rules are re-exported from ``ggmanim``. A rule ending in ``_`` is a prefix
  (``"geom_"``), any other rule an exact name (``"xlim"``). A module may also list
  extra names for the top level in a literal ``__toplevel__`` (each must be in its
  ``__all__``). ``ggmanim.__all__`` is the sorted union.

So adding ``ggmanim/geoms/area.py`` with ``__all__ = ["GeomArea", "geom_area"]``
needs no edit to any existing file: run this script and ``geom_area`` is exported
from ``ggmanim.geoms`` and ``ggmanim``.

The script only parses source files (``ast``); it never imports ggmanim, so it works
even when an ``__init__.py`` is broken by a merge conflict. Output is deterministic
and formatted the way ``ruff format`` / ruff's isort rules want it.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = "ggmanim"

START = "# -- generated exports (scripts/gen_exports.py) --"
END = "# -- end generated --"
LINE_LENGTH = 100

#: Subpackages whose ``__init__.py`` re-exports all their public modules.
PACKAGES = ("facets", "geoms", "positions", "scales", "stats")

#: ``(source, rules)`` for the top-level ``ggmanim/__init__.py``. ``source`` is a
#: module (``ggmanim/<source>.py``) or one of :data:`PACKAGES`.
TOP_LEVEL: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("aes", ("aes", "after_stat")),
    ("coords", ("coord_",)),
    ("coords_trans", ("coord_",)),
    ("facets", ("facet_",)),
    ("geoms", ("geom_",)),
    ("guides", ("guide_", "guides")),
    ("labels", ("breaks_", "label_")),
    (
        "palettes",
        (
            "BRIGHT",
            "LINETYPES",
            "MUTED",
            "PALETTES",
            "SHAPES",
            "VIBRANT",
            "Palette",
            "as_manim_color",
            "div_gradient_pal",
            "gradient_n_pal",
            "hcl_to_hex",
            "hsv_pal",
            "hue_pal",
            "resolve_palette",
            "seq_gradient_pal",
            "viridis_pal",
        ),
    ),
    ("plot", ("GGManim", "ggplot", "labs", "xlab", "ylab")),
    ("positions", ("position_",)),
    ("scales", ("scale_", "xlim", "ylim")),
    ("stats", ("stat_",)),
    ("theme", ("theme", "theme_", "element_")),
)


class ExportError(Exception):
    """A module that breaks the export rules (missing ``__all__``, duplicates ...)."""


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# -- reading -----------------------------------------------------------------------


def _literal_names(path: Path, var: str) -> list[str] | None:
    """The literal list/tuple of strings assigned to module-level ``var``."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[str] | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == var for t in targets):
            try:
                value = ast.literal_eval(node.value)  # type: ignore[arg-type]
            except ValueError as e:
                raise ExportError(f"{path}: {var} must be a literal list of strings") from e
            if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
                raise ExportError(f"{path}: {var} must be a literal list of strings")
            found = list(value)
    return found


def module_exports(path: Path) -> tuple[list[str], list[str]]:
    """``(__all__, __toplevel__)`` of the module at ``path``."""
    names = _literal_names(path, "__all__")
    if names is None:
        raise ExportError(f"{_rel(path)}: public module without __all__")
    if len(set(names)) != len(names):
        raise ExportError(f"{_rel(path)}: duplicate names in __all__")
    top = _literal_names(path, "__toplevel__") or []
    stray = sorted(set(top) - set(names))
    if stray:
        raise ExportError(f"{_rel(path)}: __toplevel__ names not in __all__: {stray}")
    return names, top


def public_modules(pkg_dir: Path) -> list[Path]:
    """``<pkg_dir>/*.py`` not starting with ``_``, sorted by name."""
    return sorted(p for p in pkg_dir.glob("*.py") if not p.name.startswith("_"))


def package_exports(root: Path, package: str) -> dict[str, tuple[list[str], list[str]]]:
    """``{module name: (__all__, __toplevel__)}`` for the public modules of ``package``;
    raises on a name exported by two modules."""
    out: dict[str, tuple[list[str], list[str]]] = {}
    owner: dict[str, str] = {}
    for path in public_modules(root / PKG / package):
        names, top = module_exports(path)
        for n in names:
            if n in owner:
                raise ExportError(
                    f"{PKG}/{package}: {n!r} is exported by both {owner[n]}.py and {path.stem}.py"
                )
            owner[n] = path.stem
        out[path.stem] = (names, top)
    return out


# -- rendering ---------------------------------------------------------------------


def _natural(s: str) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s.lower())]


def isort_key(name: str) -> tuple:
    """ruff/isort member order (``order-by-type``): CONSTANTS, Classes, functions."""
    if len(name) > 1 and name.isupper():
        kind = 0
    elif name[:1].isupper():
        kind = 1
    else:
        kind = 2
    return (kind, _natural(name), name)


def render_import(module: str, names: list[str]) -> str:
    names = sorted(names, key=isort_key)
    one = f"from .{module} import {', '.join(names)}"
    if len(one) <= LINE_LENGTH:
        return one
    body = "".join(f"    {n},\n" for n in names)
    return f"from .{module} import (\n{body})"


def render_block(imports: dict[str, list[str]]) -> str:
    """The generated block (markers included) for ``{module: names}``."""
    lines = [START]
    exported: list[str] = []
    for module in sorted(imports, key=_natural):
        names = imports[module]
        if names:
            lines.append(render_import(module, names))
            exported.extend(names)
    if exported:
        lines.append("")
        lines.append("__all__ = [")
        lines.extend(f'    "{n}",' for n in sorted(exported))
        lines.append("]")
    else:
        lines.append("__all__: list[str] = []")
    lines.append(END)
    return "\n".join(lines) + "\n"


def _matches(name: str, rules: tuple[str, ...]) -> bool:
    return any(name.startswith(r) if r.endswith("_") else name == r for r in rules)


def top_level_imports(root: Path) -> dict[str, list[str]]:
    """``{source: names}`` re-exported from ``ggmanim`` (see :data:`TOP_LEVEL`)."""
    out: dict[str, list[str]] = {}
    owner: dict[str, str] = {}
    for source, rules in TOP_LEVEL:
        if source in PACKAGES:
            modules = package_exports(root, source).values()
        else:
            modules = [module_exports(root / PKG / f"{source}.py")]
        names: list[str] = []
        for all_, top in modules:
            names.extend(n for n in all_ if _matches(n, rules) or n in top)
        for n in names:
            if n in owner:
                raise ExportError(
                    f"{PKG}: {n!r} would be exported from both {owner[n]} and {source}"
                )
            owner[n] = source
        out[source] = names
    return out


def expected_blocks(root: Path = ROOT) -> dict[Path, str]:
    """``{__init__.py path: generated block}`` for every managed file."""
    blocks = {}
    for package in PACKAGES:
        imports = {m: names for m, (names, _) in package_exports(root, package).items()}
        blocks[root / PKG / package / "__init__.py"] = render_block(imports)
    blocks[root / PKG / "__init__.py"] = render_block(top_level_imports(root))
    return blocks


def splice(text: str, block: str, path: Path) -> str:
    """``text`` with everything from the first :data:`START` line to the last
    :data:`END` line replaced by ``block``."""
    lines = text.splitlines(keepends=True)
    starts = [i for i, ln in enumerate(lines) if ln.strip() == START]
    ends = [i for i, ln in enumerate(lines) if ln.strip() == END]
    if not starts or not ends or ends[-1] < starts[0]:
        raise ExportError(f"{_rel(path)}: missing '{START}' / '{END}' markers")
    return "".join(lines[: starts[0]]) + block + "".join(lines[ends[-1] + 1 :])


# -- main --------------------------------------------------------------------------


def run(check: bool = False, root: Path = ROOT, out=sys.stdout) -> int:
    """Regenerate (or with ``check``, verify) every block. Returns the exit code."""
    try:
        blocks = expected_blocks(root)
        stale = []
        for path, block in blocks.items():
            old = path.read_text()
            new = splice(old, block, path)
            if new == old:
                continue
            rel = path.relative_to(root)
            stale.append(rel)
            if check:
                out.writelines(
                    difflib.unified_diff(
                        old.splitlines(keepends=True),
                        new.splitlines(keepends=True),
                        f"a/{rel}",
                        f"b/{rel}",
                    )
                )
            else:
                path.write_text(new)
                print(f"updated {rel}", file=out)
    except ExportError as e:
        print(f"gen_exports: error: {e}", file=out)
        return 2
    if check and stale:
        names = ", ".join(str(s) for s in stale)
        print(f"\ngen_exports: stale export blocks in {names}", file=out)
        print("run `python scripts/gen_exports.py` to regenerate them", file=out)
        return 1
    if not stale:
        print("gen_exports: export blocks are up to date", file=out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="don't write; exit 1 if any block is stale"
    )
    args = parser.parse_args(argv)
    return run(check=args.check)


if __name__ == "__main__":
    sys.exit(main())
