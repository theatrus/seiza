#!/usr/bin/env bash
# Build Seiza's offline Gaia DR3 photometry catalog (stars-gaia-photometry.bin)
# for colour calibration, from the ESA Gaia archive.
#
#   scripts/build-gaia-photometry.sh [work-dir] [max-mag]
#
# Downloads G, BP, RP and RUWE for every Gaia DR3 source to max-mag (default
# 15) in resumable chunks, then builds the catalog into the work directory
# (default ./gaia-photometry). Rerun after an interruption: finished chunks are
# kept. Run it in the background with, for example,
#
#   nohup scripts/build-gaia-photometry.sh ~/gaia-photometry > gaia.log 2>&1 &
#
# ARCHIVE=gavo fetches from the GAVO mirror in Heidelberg instead of ESA's
# archive, for when ESA's is slow or down; both carry the same columns.
# ARCHIVE=any alternates between them, round by round.
#
# Install the result by copying stars-gaia-photometry.bin into a Seiza catalog
# directory, or pass it to `seiza color-calibrate --gaia-catalog`.
set -euo pipefail

work=${1:-gaia-photometry}
max_mag=${2:-15}
seiza=${SEIZA:-seiza}
archive=${ARCHIVE:-esa}
mkdir -p "$work/chunks"

echo "$(date -u +%FT%TZ) downloading Gaia DR3 photometry to G <= $max_mag from $archive into $work/chunks"
# Finished chunks are kept, so a round that gives up on one chunk (the
# archives have bad minutes) resumes where it stopped after a pause.
for round in $(seq 1 "${ROUNDS:-20}"); do
  source=$archive
  if [ "$archive" = any ]; then
    if [ $((round % 2)) -eq 1 ]; then source=esa; else source=gavo; fi
  fi
  if "$seiza" download-data gaia-photometry --output "$work/chunks" --max-mag "$max_mag" \
    --archive "$source"; then
    break
  fi
  if [ "$round" -eq "${ROUNDS:-20}" ]; then
    echo "$(date -u +%FT%TZ) giving up after $round rounds; rerun to resume" >&2
    exit 1
  fi
  echo "$(date -u +%FT%TZ) round $round stopped; resuming in 5 minutes" >&2
  sleep 300
done

echo "$(date -u +%FT%TZ) building $work/stars-gaia-photometry.bin"
"$seiza" build-data gaia-photometry --input "$work/chunks" \
  --output "$work/stars-gaia-photometry.bin.partial" --max-mag "$max_mag"
mv "$work/stars-gaia-photometry.bin.partial" "$work/stars-gaia-photometry.bin"
echo "$(date -u +%FT%TZ) done: $(du -h "$work/stars-gaia-photometry.bin" | cut -f1)"
