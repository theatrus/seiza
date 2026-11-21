#!/usr/bin/env bash
# Fetch Microsoft's ONNX Runtime for this system, from PSF Guard's copy or
# Microsoft's own release, check it against its pinned hash, and lay out what
# a package ships in seiza-cli/onnxruntime under names that do not change with
# the version. Prints the library's path.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=packaging/onnxruntime.env
. "$root/packaging/onnxruntime.env"
version=$ONNXRUNTIME_VERSION

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)
    name=onnxruntime-linux-x64-$version
    archive=$name.tgz
    sum=$ONNXRUNTIME_SHA256_LINUX_X64
    library=libonnxruntime.so
    source_library=libonnxruntime.so.$version
    extras=(libonnxruntime_providers_shared.so)
    ;;
  MINGW*-x86_64 | MSYS*-x86_64)
    name=onnxruntime-win-x64-$version
    archive=$name.zip
    sum=$ONNXRUNTIME_SHA256_WIN_X64
    library=onnxruntime.dll
    source_library=onnxruntime.dll
    extras=(onnxruntime_providers_shared.dll)
    ;;
  Darwin-arm64)
    name=onnxruntime-osx-arm64-$version
    archive=$name.tgz
    sum=$ONNXRUNTIME_SHA256_OSX_ARM64
    library=libonnxruntime.dylib
    source_library=libonnxruntime.$version.dylib
    extras=()
    ;;
  *)
    echo "no pinned ONNX Runtime for $(uname -s) $(uname -m)" >&2
    exit 1
    ;;
esac

# Microsoft's repository has returned 404 since about 2026-11-18, so PSF
# Guard's release serves a copy of each archive, byte for byte. Microsoft's
# address stays second in case it comes back. Either must match the pin.
sources=(
  "https://github.com/theatrus/psf-guard/releases/download/onnxruntime-$version"
  "https://github.com/microsoft/onnxruntime/releases/download/v$version"
)

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
fetched=
for source in "${sources[@]}"; do
  if ! curl -fsSL --retry 3 -o "$work/$archive" "$source/$archive"; then
    echo "could not download $archive from $source" >&2
    continue
  fi
  found=$(sha256sum "$work/$archive" 2>/dev/null || shasum -a 256 "$work/$archive")
  if [ "${found%% *}" = "$sum" ]; then
    fetched=1
    break
  fi
  echo "$archive from $source hashes to ${found%% *}, not the pinned $sum" >&2
done
if [ -z "$fetched" ]; then
  echo "no source served the pinned $archive" >&2
  exit 1
fi
case "$archive" in
  # Most of the Windows archive is a 429 MB debug database nothing ships.
  *.zip) unzip -q "$work/$archive" "$name/lib/*.dll" "$name/LICENSE" "$name/ThirdPartyNotices.txt" -d "$work" ;;
  *) tar xzf "$work/$archive" -C "$work" ;;
esac

target=$root/seiza-cli/onnxruntime
rm -rf "$target"
mkdir -p "$target"
cp "$work/$name/lib/$source_library" "$target/$library"
for extra in ${extras[@]+"${extras[@]}"}; do
  cp "$work/$name/lib/$extra" "$target/$extra"
done
# Renamed, because a package puts these beside Seiza's own LICENSE.
cp "$work/$name/LICENSE" "$target/LICENSE-onnxruntime.txt"
cp "$work/$name/ThirdPartyNotices.txt" "$target/ThirdPartyNotices-onnxruntime.txt"

if command -v cygpath > /dev/null; then
  cygpath -w "$target/$library"
else
  echo "$target/$library"
fi
