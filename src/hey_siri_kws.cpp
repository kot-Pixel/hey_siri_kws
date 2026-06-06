#include "hey_siri_kws.h"

#include <cstring>
#include <memory>

#include "flex_delegate_loader.h"
#include "tensorflow/lite/c/c_api.h"

extern const unsigned char g_hey_siri_model_data[];
extern const size_t g_hey_siri_model_data_len;

namespace {

constexpr int kFrameSamples = 320;
constexpr int kNumLabels = 3;

struct KwsEngineImpl {
  std::unique_ptr<TfLiteDelegate, decltype(&flex_delegate_destroy)> flex_delegate{
      nullptr, flex_delegate_destroy};
  TfLiteModel* model = nullptr;
  TfLiteInterpreter* interpreter = nullptr;
  float last_logits[kNumLabels] = {0.f, 0.f, 0.f};
};

bool InitInterpreter(KwsEngineImpl* engine) {
  engine->model = TfLiteModelCreate(g_hey_siri_model_data, g_hey_siri_model_data_len);
  if (engine->model == nullptr) {
    return false;
  }

  flex_delegate_init();
  TfLiteDelegate* flex = flex_delegate_create();
  if (flex == nullptr) {
    return false;
  }
  engine->flex_delegate.reset(flex);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, engine->flex_delegate.get());
  engine->interpreter = TfLiteInterpreterCreate(engine->model, options);
  TfLiteInterpreterOptionsDelete(options);
  if (engine->interpreter == nullptr) {
    return false;
  }

  if (TfLiteInterpreterAllocateTensors(engine->interpreter) != kTfLiteOk) {
    return false;
  }

  return true;
}

bool FeedFrame(KwsEngineImpl* engine, const float* pcm_320) {
  TfLiteTensor* input = TfLiteInterpreterGetInputTensor(engine->interpreter, 0);
  if (input == nullptr) {
    return false;
  }

  float* input_data = reinterpret_cast<float*>(TfLiteTensorData(input));
  if (input_data == nullptr) {
    return false;
  }

  std::memcpy(input_data, pcm_320, kFrameSamples * sizeof(float));
  if (TfLiteInterpreterInvoke(engine->interpreter) != kTfLiteOk) {
    return false;
  }

  const TfLiteTensor* output = TfLiteInterpreterGetOutputTensor(engine->interpreter, 0);
  if (output == nullptr || TfLiteTensorData(output) == nullptr) {
    return false;
  }

  const float* out = reinterpret_cast<const float*>(TfLiteTensorData(output));
  std::memcpy(engine->last_logits, out, kNumLabels * sizeof(float));
  return true;
}

KwsEngineImpl* ToImpl(KwsEngine* engine) {
  return reinterpret_cast<KwsEngineImpl*>(engine);
}

}  // namespace

extern "C" {

KwsEngine* kws_create(void) {
  auto* engine = new (std::nothrow) KwsEngineImpl();
  if (engine == nullptr || !InitInterpreter(engine)) {
    delete engine;
    return nullptr;
  }
  return reinterpret_cast<KwsEngine*>(engine);
}

void kws_destroy(KwsEngine* engine) {
  if (engine == nullptr) {
    return;
  }
  auto* impl = ToImpl(engine);
  if (impl->interpreter != nullptr) {
    TfLiteInterpreterDelete(impl->interpreter);
  }
  if (impl->model != nullptr) {
    TfLiteModelDelete(impl->model);
  }
  delete impl;
}

int kws_feed_pcm_f32(KwsEngine* engine, const float* pcm_320) {
  if (engine == nullptr || pcm_320 == nullptr) {
    return -1;
  }
  return FeedFrame(ToImpl(engine), pcm_320) ? 0 : -1;
}

int kws_feed_pcm_i16(KwsEngine* engine, const int16_t* pcm_320) {
  if (engine == nullptr || pcm_320 == nullptr) {
    return -1;
  }
  float pcm_f32[kFrameSamples];
  for (int i = 0; i < kFrameSamples; ++i) {
    pcm_f32[i] = static_cast<float>(pcm_320[i]) / 32768.0f;
  }
  return FeedFrame(ToImpl(engine), pcm_f32) ? 0 : -1;
}

int kws_get_logits(KwsEngine* engine, float out_logits[3]) {
  if (engine == nullptr || out_logits == nullptr) {
    return -1;
  }
  std::memcpy(out_logits, ToImpl(engine)->last_logits, kNumLabels * sizeof(float));
  return 0;
}

int kws_get_top_label(KwsEngine* engine) {
  if (engine == nullptr) {
    return -1;
  }
  const float* logits = ToImpl(engine)->last_logits;
  int best = 0;
  float best_score = logits[0];
  for (int i = 1; i < kNumLabels; ++i) {
    if (logits[i] > best_score) {
      best_score = logits[i];
      best = i;
    }
  }
  return best;
}

float kws_get_label_score(KwsEngine* engine, int label) {
  if (engine == nullptr || label < 0 || label >= kNumLabels) {
    return 0.f;
  }
  return ToImpl(engine)->last_logits[label];
}

}  // extern "C"
