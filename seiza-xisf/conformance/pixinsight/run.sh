#!/usr/bin/env bash
# Two-way XISF conformance run against an installed PixInsight.
#
#   run.sh [work_dir]
#
# PIXINSIGHT points at the PixInsight install (default ~/PixInsight). The run
# needs xvfb-run when no display is available.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
work=${1:-$(mktemp -d -t seiza-xisf-pixinsight.XXXXXX)}
pixinsight=${PIXINSIGHT:-$HOME/PixInsight}

rm -rf "$work/from_pi" "$work/to_pi"
mkdir -p "$work/from_pi" "$work/to_pi"
for script in gen verify; do
   sed "s|__BASE__|$work|g" "$here/$script.js" > "$work/$script.js"
done

run_pixinsight() {
   local launcher=()
   if [ -z "${DISPLAY:-}" ]; then
      launcher=(xvfb-run -a)
   fi
   "${launcher[@]}" "$pixinsight/bin/PixInsight.sh" -n=7 --automation-mode --no-splash \
      --no-startup-check-updates --no-startup-gui-messages -r="$1" --force-exit \
      > "$work/$(basename "$1" .js).log" 2>&1
}

echo "== PixInsight writes the test matrix"
run_pixinsight "$work/gen.js"
grep -v '^ok' "$work/from_pi/log.txt" || true

echo "== seiza reads it and writes files for PixInsight"
cargo run --quiet --release --manifest-path "$repo/Cargo.toml" -p seiza-xisf \
   --example pixinsight_probe -- "$work/from_pi" "$work/to_pi" > "$work/probe.txt"
grep -E '^FAIL|^read checks' "$work/probe.txt"

echo "== PixInsight reads seiza's files"
run_pixinsight "$work/verify.js"
fails=$(grep -c '^FAIL\|^EXC' "$work/verify.txt" || true)
total=$(grep -c '^ok\|^FAIL\|^EXC' "$work/verify.txt" || true)
grep -A20 '^FAIL\|^EXC' "$work/verify.txt" | grep -v '^ok' || true
echo "PixInsight checks: $((total - fails)) passed, $fails failed"
echo "Details in $work"
