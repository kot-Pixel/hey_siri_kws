#ifndef FLEX_DELEGATE_LOADER_H_
#define FLEX_DELEGATE_LOADER_H_

#include "tensorflow/lite/core/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif

int flex_delegate_init(void);
TfLiteDelegate* flex_delegate_create(void);
void flex_delegate_destroy(TfLiteDelegate* delegate);

#ifdef __cplusplus
}
#endif

#endif  // FLEX_DELEGATE_LOADER_H_
