#!/bin/bash
set -euo pipefail
BUILD="${1:-$HOME/tflite_static_build_arm64-v8a}"
OUT="/tmp/tflite_deps.txt"
ninja -C "$BUILD" -t query libtensorflow-lite.a > "$OUT"

python3 - <<'PY'
import re, sys
text=open("/tmp/tflite_deps.txt").read()
libs=set(re.findall(r'lib[\w+\-.]+\.a', text))
for name in sorted(libs):
    print(name)
print(f"# total {len(libs)}", file=sys.stderr)
PY
