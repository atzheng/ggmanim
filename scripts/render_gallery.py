"""Render every example scene for the docs gallery and write ``docs/gallery/index.md``.

Usage::

    python scripts/render_gallery.py               # render all scenes + write the page
    python scripts/render_gallery.py -j 4          # at most 4 manim processes at once
    python scripts/render_gallery.py --only polar  # scenes whose "module.Scene" contains "polar"
    python scripts/render_gallery.py --index-only  # just write the page, no rendering

Every ``Scene`` subclass defined in ``examples/*.py`` (found by parsing, not
importing, the files; names starting with ``_`` are skipped) is rendered in its
own ``manim`` process, in parallel, into a throwaway media directory; the result
is copied to ``docs/gallery/img/<module>/<Scene>.png`` (or ``.gif``). Stills are
the last frame at ``-ql`` (``-s``). The scenes listed in :data:`ANIMATED` are
animation showcases and are rendered as ``-ql`` GIFs at :data:`GIF_FPS` instead.

Everything under ``docs/gallery/`` is build output and ignored by version
control: the docs workflow runs this script before ``mkdocs build``. The page is
generated rather than committed because it repeats the example sources
(thousands of lines that would go stale and conflict between branches adding
examples), and a docs build needs the rendered images anyway.
"""

from __future__ import annotations

import argparse
import ast
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
GALLERY = ROOT / "docs" / "gallery"
INDEX = GALLERY / "index.md"
IMG_DIR = GALLERY / "img"
SOURCE_URL = "https://github.com/atzheng/ggmanim/blob/main/examples/"

#: Scenes rendered as GIFs: module -> scene names (``None``: every scene in it).
ANIMATED: dict[str, frozenset[str] | None] = {
    "animate_create": None,
    "live_plot": None,
    "transitions": None,
    "coord_trans": frozenset({"LineToCurve"}),
    "polar": frozenset({"BarsToPie"}),
    "gallery_facets": frozenset({"PanelReveal"}),
}
GIF_FPS = 10
#: Seconds before one manim process is killed.
TIMEOUT = 600


@dataclass(frozen=True)
class GalleryScene:
    """One scene of one example file."""

    module: str
    name: str
    path: Path
    doc: str
    source: str
    animated: bool

    @property
    def key(self) -> str:
        return f"{self.module}.{self.name}"

    @property
    def anchor(self) -> str:
        return f"{self.module}-{self.name}".lower().replace("_", "-")

    @property
    def image(self) -> str:
        """Image path relative to ``docs/gallery/``."""
        return f"img/{self.module}/{self.name}.{'gif' if self.animated else 'png'}"


@dataclass(frozen=True)
class Example:
    """One ``examples/*.py`` file and its scenes."""

    module: str
    path: Path
    summary: str
    scenes: tuple[GalleryScene, ...]


# --------------------------------------------------------------------------- discovery


def _first_paragraph(doc: str | None) -> str:
    if not doc:
        return ""
    para = doc.strip().split("\n\n", 1)[0]
    return " ".join(line.strip() for line in para.splitlines())


def _base_names(node: ast.ClassDef) -> list[str]:
    names = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def _defined_names(node: ast.stmt) -> list[str]:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def _used_names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _start_line(node: ast.stmt) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(d.lineno for d in decorators)])


def scene_source(tree: ast.Module, lines: Sequence[str], cls: ast.ClassDef) -> str:
    """The scene class plus the top-level definitions of its file it uses (transitively).

    Imports are left out; definitions keep their source order.
    """
    defs: dict[str, ast.stmt] = {}
    for node in tree.body:
        for name in _defined_names(node):
            defs[name] = node
    needed: dict[int, ast.stmt] = {id(cls): cls}
    todo: list[ast.stmt] = [cls]
    while todo:
        for name in sorted(_used_names(todo.pop())):
            dep = defs.get(name)
            if dep is not None and id(dep) not in needed:
                needed[id(dep)] = dep
                todo.append(dep)
    chunks = []
    for node in sorted(needed.values(), key=_start_line):
        end = node.end_lineno or node.lineno
        chunks.append("\n".join(lines[_start_line(node) - 1 : end]).rstrip())
    return "\n\n\n".join(chunks)


def _is_animated(module: str, name: str) -> bool:
    if module not in ANIMATED:
        return False
    names = ANIMATED[module]
    return names is None or name in names


def discover_example(path: Path) -> Example:
    """Parse one example file (without importing it) and list its public scenes."""
    text = path.read_text()
    tree = ast.parse(text, filename=str(path))
    lines = text.splitlines()
    module = path.stem
    scene_classes = {"Scene"}
    scenes = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not scene_classes.intersection(_base_names(node)):
            continue
        scene_classes.add(node.name)
        if node.name.startswith("_"):
            continue
        scenes.append(
            GalleryScene(
                module=module,
                name=node.name,
                path=path,
                doc=_first_paragraph(ast.get_docstring(node)),
                source=scene_source(tree, lines, node),
                animated=_is_animated(module, node.name),
            )
        )
    return Example(module, path, _first_paragraph(ast.get_docstring(tree)), tuple(scenes))


def discover(examples_dir: Path = EXAMPLES) -> list[Example]:
    """Every ``*.py`` in ``examples_dir`` that defines at least one public scene."""
    found = (discover_example(p) for p in sorted(examples_dir.glob("*.py")))
    return [ex for ex in found if ex.scenes]


# --------------------------------------------------------------------------- page


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line if line.strip() else "" for line in text.splitlines())


def render_index(examples: Iterable[Example]) -> str:
    """The gallery page (markdown) for ``examples``; depends on the sources only."""
    examples = list(examples)
    n_scenes = sum(len(ex.scenes) for ex in examples)
    n_gifs = sum(sc.animated for ex in examples for sc in ex.scenes)
    out = [
        "<!-- Generated by scripts/render_gallery.py from examples/*.py; do not edit. -->",
        "",
        "# Gallery",
        "",
        f"Every scene in [`examples/`]({SOURCE_URL}): {n_scenes} scenes from "
        f"{len(examples)} files. Stills show the last frame (`manim -ql -s`); the "
        f"{n_gifs} animation showcases are low-quality GIFs ({GIF_FPS} fps). Each "
        "source block shows the scene class and the top-level helpers of its file "
        "that it uses; follow the file link for the imports and data.",
        "",
        "This page and its images are generated, not checked in; build them locally with",
        "",
        "```bash",
        "python scripts/render_gallery.py",
        "mkdocs serve",
        "```",
        "",
    ]
    for ex in examples:
        out.append(f"[{ex.module}.py](#{ex.module.replace('_', '-')})")
        out.append("")
        for sc in ex.scenes:
            out.append(
                f'[![{sc.key}]({sc.image}){{ width="200" loading=lazy }}](#{sc.anchor} "{sc.key}")'
            )
        out.append("")
    for ex in examples:
        out += [f"## {ex.module}.py {{ #{ex.module.replace('_', '-')} }}", ""]
        if ex.summary:
            out += [ex.summary, ""]
        out += [f"Source: [`examples/{ex.path.name}`]({SOURCE_URL}{ex.path.name})", ""]
        for sc in ex.scenes:
            flags = f"--format gif --fps {GIF_FPS}" if sc.animated else "-s"
            out += [f"### {sc.name} {{ #{sc.anchor} }}", ""]
            if sc.doc:
                out += [sc.doc, ""]
            out += [
                f"![{sc.key}]({sc.image}){{ loading=lazy }}",
                "",
                f'???+ example "`manim -ql {flags} examples/{ex.path.name} {sc.name}`"',
                "",
                "    ```python",
                _indent(sc.source),
                "    ```",
                "",
            ]
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------- render


@dataclass(frozen=True)
class RenderResult:
    scene: GalleryScene
    ok: bool
    seconds: float
    log: str


def manim_args(scene: GalleryScene, media_dir: Path) -> list[str]:
    """Arguments after the manim executable for one scene."""
    fmt = ["--format", "gif", "--fps", str(GIF_FPS)] if scene.animated else ["-s"]
    return [
        "-ql",
        *fmt,
        "--disable_caching",
        "--media_dir",
        str(media_dir),
        str(scene.path),
        scene.name,
    ]


def render_scene(scene: GalleryScene, manim: Sequence[str], img_dir: Path) -> RenderResult:
    """Render one scene in its own process and copy the image into ``img_dir``."""
    start = time.perf_counter()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    with tempfile.TemporaryDirectory(prefix="ggm-gallery-") as tmp:
        media = Path(tmp)
        try:
            proc = subprocess.run(
                [*manim, *manim_args(scene, media)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
            )
            log, ok = proc.stdout + proc.stderr, proc.returncode == 0
        except subprocess.TimeoutExpired:
            log, ok = f"timed out after {TIMEOUT}s", False
        suffix = ".gif" if scene.animated else ".png"
        outputs = [
            p for p in media.rglob(f"{scene.name}*{suffix}") if "partial_movie_files" not in p.parts
        ]
        if ok and len(outputs) != 1:
            ok = False
            log += f"\nexpected one {suffix} output, found {[str(p) for p in outputs]}"
        if ok:
            target = img_dir / scene.image.removeprefix("img/")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(outputs[0], target)
    return RenderResult(scene, ok, time.perf_counter() - start, log)


def render_all(
    scenes: Sequence[GalleryScene], manim: Sequence[str], jobs: int, img_dir: Path = IMG_DIR
) -> list[RenderResult]:
    """Render ``scenes`` with ``jobs`` processes at once, GIFs first (they are slowest)."""
    ordered = sorted(scenes, key=lambda s: not s.animated)
    results = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(render_scene, s, manim, img_dir) for s in ordered]
        for fut in as_completed(futures):
            res = fut.result()
            status = "ok  " if res.ok else "FAIL"
            print(f"{status} {res.seconds:6.1f}s  {res.scene.key}", flush=True)
            results.append(res)
    return results


# --------------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--index-only", action="store_true", help="write index.md, no render")
    parser.add_argument("--only", help="render only scenes whose 'module.Scene' contains this")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 2)
    parser.add_argument(
        "--manim",
        default=None,
        help="manim command (default: the current Python's `-m manim`)",
    )
    args = parser.parse_args(argv)

    examples = discover()
    page = render_index(examples)
    GALLERY.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(page)
    print(f"wrote {INDEX.relative_to(ROOT)}")
    if args.index_only:
        return 0

    scenes = [sc for ex in examples for sc in ex.scenes]
    if args.only:
        scenes = [sc for sc in scenes if args.only in sc.key]
    elif IMG_DIR.exists():
        shutil.rmtree(IMG_DIR)  # full render: drop images of removed scenes
    manim = shlex.split(args.manim) if args.manim else [sys.executable, "-m", "manim"]
    start = time.perf_counter()
    results = render_all(scenes, manim, max(1, args.jobs))
    wall = time.perf_counter() - start
    failed = [r for r in results if not r.ok]
    for res in failed:
        print(f"\n===== {res.scene.key} failed =====\n{res.log[-4000:]}", file=sys.stderr)
    n_gif = sum(r.scene.animated for r in results)
    print(
        f"rendered {len(results) - len(failed)}/{len(results)} scenes "
        f"({n_gif} GIFs) in {wall:.0f}s wall, {sum(r.seconds for r in results):.0f}s total, "
        f"{args.jobs} jobs"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
