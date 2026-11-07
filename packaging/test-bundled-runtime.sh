#!/usr/bin/env bash
# Run a tiny model on an installed package's own ONNX Runtime, as a fresh
# machine would, with nothing telling Seiza where the library is.
# usage: test-bundled-runtime.sh BINARY LIBRARY
set -euo pipefail

binary=$1
library=$2
fixtures=$(cd "$(dirname "${BASH_SOURCE[0]}")/../seiza-model/tests/fixtures" && pwd)
unset SEIZA_ONNXRUNTIME

test -f "$library"
report=$("$binary" model backends)
printf '%s\n' "$report"
if ! grep -Fq " at $library" <<< "$report"; then
  echo "seiza did not load $library" >&2
  exit 1
fi

out=$(mktemp -d)
"$binary" model denoise "$fixtures/tiny-input.fits" -o "$out/clean.fits" --model "$fixtures/tiny" --backend cpu
test -s "$out/clean.fits"
echo "a tiny model ran on $library"
