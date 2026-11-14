"""Compare two trees of rendered PNGs pixel by pixel.

Used to check that a performance change leaves the gallery unchanged: render
every example before and after (``scripts/render_examples.sh <media_dir>``) and
compare the two ``<media_dir>/images`` trees::

    python benchmarks/compare_images.py before/images after/images

Every ``*.png`` under ``BEFORE`` is matched by its relative path under ``AFTER``.
Prints one line per image (``identical`` or the number of differing pixels and
the largest channel difference) and exits 1 if any image differs, is missing or
has a different size (``--tolerance`` allows a maximum channel difference).
Needs only numpy and Pillow (both manim dependencies).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"), dtype=np.int16)


def compare(before: Path, after: Path, tolerance: int = 0) -> tuple[list[str], bool]:
    """``(report lines, all equal)`` for the PNG trees ``before`` and ``after``."""
    lines: list[str] = []
    ok = True
    olds = sorted(p.relative_to(before) for p in before.rglob("*.png"))
    news = {p.relative_to(after) for p in after.rglob("*.png")}
    if not olds:
        return [f"no PNGs under {before}"], False
    for rel in olds:
        if rel not in news:
            lines.append(f"MISSING    {rel}")
            ok = False
            continue
        a, b = load(before / rel), load(after / rel)
        if a.shape != b.shape:
            lines.append(f"SIZE       {rel}: {a.shape[:2]} vs {b.shape[:2]}")
            ok = False
            continue
        diff = np.abs(a - b).max(axis=2)
        n, worst = int((diff > tolerance).sum()), int(diff.max())
        if n:
            lines.append(f"DIFFERENT  {rel}: {n} pixel(s), max channel diff {worst}")
            ok = False
        else:
            lines.append(f"identical  {rel}" + (f" (max diff {worst})" if worst else ""))
    for rel in sorted(news - set(olds)):
        lines.append(f"NEW        {rel}")
    return lines, ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--tolerance", type=int, default=0, help="allowed channel difference")
    args = ap.parse_args(argv)
    lines, ok = compare(args.before, args.after, args.tolerance)
    print("\n".join(lines))
    same = sum(1 for line in lines if line.startswith("identical"))
    print(f"\n{same}/{len(lines)} image(s) identical")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
