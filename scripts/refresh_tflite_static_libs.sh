#!/bin/bash
# Re-copy minimal TFLite static libs from existing WSL build dirs (no recompile).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="$PROJECT_ROOT/third_party/tflite-static"

refresh_abi() {
  local abi="$1"
  local build_dir="$HOME/tflite_static_build_${abi}"
  local out_dir="$OUT_ROOT/$abi"
  local lib_dir="$out_dir/lib"
  if [[ ! -d "$build_dir" ]]; then
    echo "Skip $abi: missing $build_dir (run build_tflite_static.sh first)" >&2
    return 1
  fi
  rm -rf "$lib_dir"
  mkdir -p "$lib_dir"
  mapfile -t needed < <(bash "$PROJECT_ROOT/scripts/list_tflite_link_libs.sh" "$build_dir")
  for name in "${needed[@]}"; do
    cp -f "$(find "$build_dir" -name "$name" -type f | head -1)" "$lib_dir/"
  done
  printf '%s\n' "${needed[@]}" > "$out_dir/lib/libs.txt"
  echo "$abi: ${#needed[@]} libs"
}

for abi in arm64-v8a armeabi-v7a; do
  refresh_abi "$abi"
done
