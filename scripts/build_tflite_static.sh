#!/bin/bash
# Build static TensorFlow Lite (C API) for Android ABIs in WSL.
# Output: third_party/tflite-static/<abi>/lib/*.a + include/
set -euo pipefail

TF_TAG="${TF_TAG:-v2.14.0}"
API_LEVEL="${API_LEVEL:-24}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="${TF_SRC_DIR:-$HOME/tensorflow_src}"
OUT_ROOT="$PROJECT_ROOT/third_party/tflite-static"
LINUX_NDK="${LINUX_NDK:-$HOME/android-ndk-r27d}"

resolve_ndk() {
  if [[ -n "${ANDROID_NDK_HOME:-}" && -f "${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake" ]]; then
    local prebuilt
    prebuilt="$(ls -d "${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/"* 2>/dev/null | head -1 || true)"
    if [[ "$prebuilt" == *linux-x86_64* && -x "${prebuilt}/bin/clang" ]]; then
      echo "$ANDROID_NDK_HOME"
      return
    fi
    echo "ANDROID_NDK_HOME is a Windows NDK; using Linux NDK at $LINUX_NDK instead." >&2
  fi

  if [[ ! -f "$LINUX_NDK/build/cmake/android.toolchain.cmake" ]]; then
    echo "Linux NDK not found at $LINUX_NDK" >&2
    echo "Downloading android-ndk-r27d-linux.zip (~660MB), one-time setup ..." >&2
    local zip="/tmp/android-ndk-r27d-linux.zip"
    wget -O "$zip" "https://dl.google.com/android/repository/android-ndk-r27d-linux.zip"
    rm -rf "$LINUX_NDK"
    unzip -q "$zip" -d "$HOME"
    rm -f "$zip"
  fi
  echo "$LINUX_NDK"
}

NDK="$(resolve_ndk)"

if [[ ! -d "$SRC_DIR/.git" ]]; then
  echo "Cloning TensorFlow $TF_TAG ..."
  git clone --depth 1 --branch "$TF_TAG" https://github.com/tensorflow/tensorflow.git "$SRC_DIR"
fi

build_abi() {
  local abi="$1"
  local build_dir="$HOME/tflite_static_build_${abi}"
  local out_dir="$OUT_ROOT/$abi"
  local lib_dir="$out_dir/lib"

  echo ""
  echo "=== Building TFLite static for $abi ==="
  rm -rf "$build_dir"
  mkdir -p "$build_dir" "$lib_dir"
  cd "$build_dir"

  local enable_xnnpack=ON
  if [[ "$abi" == "armeabi-v7a" ]]; then
    enable_xnnpack=OFF
    echo "XNNPACK disabled for $abi (avoid armeabi-v7a asm issue)" >&2
  fi

  cmake "$SRC_DIR/tensorflow/lite" \
    -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$abi" \
    -DANDROID_PLATFORM="android-$API_LEVEL" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DTFLITE_ENABLE_XNNPACK="$enable_xnnpack" \
    -DTFLITE_ENABLE_GPU=OFF \
    -DTFLITE_ENABLE_INSTALL=OFF

  cmake --build . -j"$(nproc)"

  echo "Collecting runtime static libraries (tensorflow-lite closure only) ..."
  rm -rf "$lib_dir"
  mkdir -p "$lib_dir"
  mapfile -t needed < <(bash "$PROJECT_ROOT/scripts/list_tflite_link_libs.sh" "$build_dir")
  for name in "${needed[@]}"; do
    path="$(find "$build_dir" -name "$name" -type f | head -1)"
    if [[ -z "$path" ]]; then
      echo "Missing $name in $build_dir" >&2
      exit 1
    fi
    cp -f "$path" "$lib_dir/"
  done
  printf '%s\n' "${needed[@]}" > "$out_dir/lib/libs.txt"

  echo "Installed ${#needed[@]} archives to $lib_dir"
}

mkdir -p "$OUT_ROOT/include/tensorflow/lite/c"
mkdir -p "$OUT_ROOT/include/tensorflow/lite/core/c"
mkdir -p "$OUT_ROOT/include/tensorflow/lite/core/async/c"
rsync -a --delete "$SRC_DIR/tensorflow/lite/c/" "$OUT_ROOT/include/tensorflow/lite/c/"
rsync -a "$SRC_DIR/tensorflow/lite/core/c/" "$OUT_ROOT/include/tensorflow/lite/core/c/"
rsync -a "$SRC_DIR/tensorflow/lite/core/async/c/" "$OUT_ROOT/include/tensorflow/lite/core/async/c/" 2>/dev/null || true
cp -f "$SRC_DIR/tensorflow/lite/builtin_ops.h" "$OUT_ROOT/include/tensorflow/lite/"

for abi in arm64-v8a armeabi-v7a; do
  build_abi "$abi"
done

echo ""
echo "=== Done ==="
echo "Static TFLite libs: $OUT_ROOT/<abi>/lib/"
echo "Headers:            $OUT_ROOT/include/"
