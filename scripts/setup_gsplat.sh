#!/usr/bin/env bash
# Builds build/gsplat_examples/: gsplat's examples/ (deps/gsplat submodule, left unmodified) with our changed files
# from gsplat_patch/examples/ copied over them. scripts/run_render.sh trains with build/gsplat_examples/simple_trainer.py.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$ROOT/deps/gsplat/examples/simple_trainer.py" ] || { echo "deps/gsplat is empty: git submodule update --init --recursive"; exit 1; }
rm -rf "$ROOT/build/gsplat_examples"; mkdir -p "$ROOT/build"
cp -r "$ROOT/deps/gsplat/examples" "$ROOT/build/gsplat_examples"
cp -r "$ROOT/gsplat_patch/examples/." "$ROOT/build/gsplat_examples/"
echo "build/gsplat_examples: gsplat examples + $(cd "$ROOT/gsplat_patch/examples" && find . -type f | sed 's|^\./||' | tr '\n' ' ')"
