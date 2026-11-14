#!/usr/bin/env bash
# Render a still of every scene in every examples/*.py with `manim -ql -s -a`.
#
# Usage: scripts/render_examples.sh [MEDIA_DIR] [EXTRA MANIM ARGS...]
#   MEDIA_DIR defaults to ./media; PNGs land in MEDIA_DIR/images/<example>/.
#   MANIM overrides the manim executable (default: `manim` on PATH).
#
# Each example is rendered even if an earlier one fails; the script exits 1 if
# any failed. New features add their own examples/<feature>.py (picked up here
# automatically) instead of editing an existing example.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
media_dir="${1:-media}"
shift || true
manim_bin="${MANIM:-manim}"

shopt -s nullglob
examples=(examples/*.py)
if [ ${#examples[@]} -eq 0 ]; then
  echo "no examples/*.py found" >&2
  exit 1
fi

failed=()
for example in "${examples[@]}"; do
  echo "::group::render ${example}"
  if ! "${manim_bin}" -ql -s -a --media_dir "${media_dir}" "$@" "${example}"; then
    failed+=("${example}")
  fi
  echo "::endgroup::"
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "failed to render: ${failed[*]}" >&2
  exit 1
fi
echo "rendered ${#examples[@]} example file(s) into ${media_dir}"
