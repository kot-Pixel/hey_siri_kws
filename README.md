# hey_siri_kws

Android ARM 动态库，PCM 流式关键词检测（hey_siri）。

## 交付物

编译后 `dist/<abi>/` 目录：

```
dist/arm64-v8a/
  lib/libhey_siri_kws.so
  include/hey_siri_kws.h
```

## 依赖

- Android NDK（设置 `ANDROID_NDK_HOME`）
- CMake + Ninja
- Python 3

## 编译

```powershell
cd E:\WorkSpace\Person\hey_siri_kws

# 可选：WSL 内编译静态 TFLite（减少 APK 依赖，只需 libhey_siri_kws.so）
# wsl -d Ubuntu-22.04 -- bash /mnt/e/WorkSpace/Person/hey_siri_kws/scripts/build_tflite_static.sh

# 首次（动态 TFLite 回退）：下载 TFLite 依赖
.\scripts\prepare_deps.ps1

# 编译 arm64
.\scripts\build.ps1 -Abi arm64-v8a

# 编译 armeabi-v7a
.\scripts\build.ps1 -Abi armeabi-v7a
```

若已执行 `build_tflite_static.sh`，`build.ps1` 会自动静态链 TFLite（`c++_static`），**无需**再打包 `libtensorflowlite_jni.so` / `libc++_shared.so`。

## C API

```c
#include "hey_siri_kws.h"

KwsEngine* engine = kws_create();

// 每 20ms 喂 320 个 int16 PCM 采样点 (16kHz mono)
int16_t pcm[320];
kws_feed_pcm_i16(engine, pcm);

float score = kws_get_label_score(engine, KWS_LABEL_HEY_SIRI);
int label = kws_get_top_label(engine);  // 0=silence, 1=unknown, 2=hey_siri

kws_destroy(engine);
```

## 运行时 native 依赖

**静态 TFLite（推荐，WSL 编译后）：** 只需 `libhey_siri_kws.so`。

**动态 TFLite（默认回退）：** `libhey_siri_kws.so` 还需随 APK 打包（在 `dist/<abi>/runtime/` 或 `third_party/tflite/lib/<abi>/`）：

- `libtensorflowlite_jni.so`
- `libc++_shared.so`

纯 C/C++ 接入，无需 Java 代码；库名中的 `_jni` 来自上游预编译包命名。MFCC 在库内计算，模型为纯 TFLite（无需 Flex）。

## 更新模型

替换 `model_mfcc/stream_state_internal.tflite` 后重新执行 `.\scripts\build.ps1`。
