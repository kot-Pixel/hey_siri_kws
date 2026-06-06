#include "flex_delegate_loader.h"

#include <cstdint>

// Flex delegate symbols exported by libtensorflowlite_flex_jni.so.
// Names are historical; callers pass null context pointers.
extern "C" {
void Java_org_tensorflow_lite_flex_FlexDelegate_nativeInitTensorFlow(void* env,
                                                                    void* clazz);
int64_t Java_org_tensorflow_lite_flex_FlexDelegate_nativeCreateDelegate(void* env,
                                                                        void* clazz);
void Java_org_tensorflow_lite_flex_FlexDelegate_nativeDeleteDelegate(void* env,
                                                                   void* clazz,
                                                                   int64_t delegate);
}

int flex_delegate_init(void) {
  Java_org_tensorflow_lite_flex_FlexDelegate_nativeInitTensorFlow(nullptr, nullptr);
  return 0;
}

TfLiteDelegate* flex_delegate_create(void) {
  return reinterpret_cast<TfLiteDelegate*>(
      Java_org_tensorflow_lite_flex_FlexDelegate_nativeCreateDelegate(nullptr, nullptr));
}

void flex_delegate_destroy(TfLiteDelegate* delegate) {
  if (delegate == nullptr) {
    return;
  }
  Java_org_tensorflow_lite_flex_FlexDelegate_nativeDeleteDelegate(
      nullptr, nullptr, reinterpret_cast<int64_t>(delegate));
}
