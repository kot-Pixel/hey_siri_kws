#include "hey_siri_kws.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

#include "mfcc_extractor.h"
#include "tensorflow/lite/c/c_api.h"

extern const unsigned char g_hey_siri_model_data[];
extern const size_t g_hey_siri_model_data_len;

namespace {

constexpr int kFrameSamples = KWS_FRAME_SAMPLES;
constexpr int kNumLabels = 3;
constexpr int kMfccFeatures = KWS_MFCC_FEATURES;
constexpr int kMfccFrames = KWS_MFCC_FRAMES;
constexpr int kMfccWindow = kMfccFrames * kMfccFeatures;
constexpr int kInferStride = KWS_INFER_STRIDE;
constexpr int kWakeHits = KWS_WAKE_HITS;
constexpr float kWakeThreshold = KWS_WAKE_THRESHOLD;
constexpr int kSiriLabel = KWS_LABEL_HEY_SIRI;

kws_mfcc::MfccConfig DefaultMfccConfig() {
  kws_mfcc::MfccConfig config;
  config.sample_rate = 16000;
  config.window_size_samples = 320;
  config.window_stride_samples = 320;
  config.mel_num_bins = 40;
  config.dct_num_features = 20;
  config.mel_lower_edge_hertz = 20.0f;
  config.mel_upper_edge_hertz = 7600.0f;
  config.magnitude_squared = false;
  return config;
}

struct KwsEngineImpl {
  TfLiteModel* model = nullptr;
  TfLiteInterpreter* interpreter = nullptr;
  kws_mfcc::MfccExtractor mfcc{DefaultMfccConfig()};
  float last_logits[kNumLabels] = {0.f, 0.f, 0.f};
  float last_probs[kNumLabels] = {0.f, 0.f, 0.f};
  float last_mfcc[kMfccFeatures] = {};
  float mfcc_window[kMfccWindow] = {};
  int frames_until_infer = 0;
  int wake_hits = 0;
};

bool ComputeMfcc(KwsEngineImpl* engine, const float* pcm_320, float* mfcc_out) {
  return engine->mfcc.ComputeFrame(pcm_320, mfcc_out);
}

bool InitInterpreter(KwsEngineImpl* engine) {
  engine->model = TfLiteModelCreate(g_hey_siri_model_data, g_hey_siri_model_data_len);
  if (engine->model == nullptr) {
    return false;
  }

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsSetNumThreads(options, 1);
  engine->interpreter = TfLiteInterpreterCreate(engine->model, options);
  TfLiteInterpreterOptionsDelete(options);
  if (engine->interpreter == nullptr) {
    return false;
  }

  if (TfLiteInterpreterAllocateTensors(engine->interpreter) != kTfLiteOk) {
    return false;
  }

  const TfLiteTensor* input = TfLiteInterpreterGetInputTensor(engine->interpreter, 0);
  if (input == nullptr) {
    return false;
  }
  const TfLiteType input_type = TfLiteTensorType(input);
  if (input_type == kTfLiteFloat32 &&
      TfLiteTensorByteSize(input) == kMfccWindow * sizeof(float)) {
    return true;
  }
  if ((input_type == kTfLiteInt8 || input_type == kTfLiteUInt8) &&
      TfLiteTensorByteSize(input) == kMfccWindow) {
    return true;
  }
  return false;
}

bool FeedFrame(KwsEngineImpl* engine, const float* mfcc_20) {
  std::memmove(engine->mfcc_window, engine->mfcc_window + kMfccFeatures,
               (kMfccFrames - 1) * kMfccFeatures * sizeof(float));
  std::memcpy(engine->mfcc_window + (kMfccFrames - 1) * kMfccFeatures, mfcc_20,
              kMfccFeatures * sizeof(float));

  if (engine->frames_until_infer > 0) {
    engine->frames_until_infer -= 1;
    return true;
  }
  engine->frames_until_infer = kInferStride - 1;

  TfLiteTensor* input = TfLiteInterpreterGetInputTensor(engine->interpreter, 0);
  if (input == nullptr) {
    return false;
  }

  const TfLiteType input_type = TfLiteTensorType(input);
  if (input_type == kTfLiteFloat32) {
    float* input_data = reinterpret_cast<float*>(TfLiteTensorData(input));
    if (input_data == nullptr) {
      return false;
    }
    std::memcpy(input_data, engine->mfcc_window, kMfccWindow * sizeof(float));
  } else if (input_type == kTfLiteInt8 || input_type == kTfLiteUInt8) {
    const TfLiteQuantizationParams qparams = TfLiteTensorQuantizationParams(input);
    if (qparams.scale == 0.0f) {
      return false;
    }
    for (int i = 0; i < kMfccWindow; ++i) {
      int q = static_cast<int>(std::lround(engine->mfcc_window[i] / qparams.scale) +
                               qparams.zero_point);
      if (input_type == kTfLiteInt8) {
        q = std::max(-128, std::min(127, q));
        reinterpret_cast<int8_t*>(TfLiteTensorData(input))[i] = static_cast<int8_t>(q);
      } else {
        q = std::max(0, std::min(255, q));
        reinterpret_cast<uint8_t*>(TfLiteTensorData(input))[i] = static_cast<uint8_t>(q);
      }
    }
  } else {
    return false;
  }
  if (TfLiteInterpreterInvoke(engine->interpreter) != kTfLiteOk) {
    return false;
  }

  const TfLiteTensor* output = TfLiteInterpreterGetOutputTensor(engine->interpreter, 0);
  if (output == nullptr || TfLiteTensorData(output) == nullptr) {
    return false;
  }

  const float* out = reinterpret_cast<const float*>(TfLiteTensorData(output));
  std::memcpy(engine->last_logits, out, kNumLabels * sizeof(float));

  float max_logit = engine->last_logits[0];
  for (int i = 1; i < kNumLabels; ++i) {
    max_logit = std::max(max_logit, engine->last_logits[i]);
  }
  float sum = 0.f;
  for (int i = 0; i < kNumLabels; ++i) {
    engine->last_probs[i] = std::exp(engine->last_logits[i] - max_logit);
    sum += engine->last_probs[i];
  }
  if (sum > 0.f) {
    for (int i = 0; i < kNumLabels; ++i) {
      engine->last_probs[i] /= sum;
    }
  }
  if (engine->last_probs[kSiriLabel] >= kWakeThreshold) {
    engine->wake_hits += 1;
  } else {
    engine->wake_hits = 0;
  }
  return true;
}

KwsEngineImpl* ToImpl(KwsEngine* engine) {
  return reinterpret_cast<KwsEngineImpl*>(engine);
}

}  // namespace

extern "C" {

KwsEngine* kws_create(void) {
  auto* engine = new (std::nothrow) KwsEngineImpl();
  if (engine == nullptr || !engine->mfcc.Initialize() || !InitInterpreter(engine)) {
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

int kws_compute_mfcc_f32(KwsEngine* engine, const float* pcm_320, float mfcc_20[kMfccFeatures]) {
  if (engine == nullptr || pcm_320 == nullptr || mfcc_20 == nullptr) {
    return -1;
  }
  auto* impl = ToImpl(engine);
  if (!ComputeMfcc(impl, pcm_320, mfcc_20)) {
    return -1;
  }
  std::memcpy(impl->last_mfcc, mfcc_20, kMfccFeatures * sizeof(float));
  return 0;
}

int kws_compute_mfcc_i16(KwsEngine* engine, const int16_t* pcm_320, float mfcc_20[kMfccFeatures]) {
  if (engine == nullptr || pcm_320 == nullptr || mfcc_20 == nullptr) {
    return -1;
  }
  float pcm_f32[kFrameSamples];
  for (int i = 0; i < kFrameSamples; ++i) {
    pcm_f32[i] = static_cast<float>(pcm_320[i]) / 32768.0f;
  }
  return kws_compute_mfcc_f32(engine, pcm_f32, mfcc_20);
}

int kws_get_last_mfcc(KwsEngine* engine, float mfcc_20[kMfccFeatures]) {
  if (engine == nullptr || mfcc_20 == nullptr) {
    return -1;
  }
  std::memcpy(mfcc_20, ToImpl(engine)->last_mfcc, kMfccFeatures * sizeof(float));
  return 0;
}

void kws_reset_mfcc(KwsEngine* engine) {
  if (engine == nullptr) {
    return;
  }
  auto* impl = ToImpl(engine);
  impl->mfcc.Reset();
  std::memset(impl->mfcc_window, 0, sizeof(impl->mfcc_window));
  impl->frames_until_infer = 0;
  impl->wake_hits = 0;
  std::memset(impl->last_probs, 0, sizeof(impl->last_probs));
}

int kws_feed_pcm_f32(KwsEngine* engine, const float* pcm_320) {
  if (engine == nullptr || pcm_320 == nullptr) {
    return -1;
  }
  auto* impl = ToImpl(engine);
  if (!ComputeMfcc(impl, pcm_320, impl->last_mfcc)) {
    return -1;
  }
  return FeedFrame(impl, impl->last_mfcc) ? 0 : -1;
}

int kws_feed_pcm_i16(KwsEngine* engine, const int16_t* pcm_320) {
  if (engine == nullptr || pcm_320 == nullptr) {
    return -1;
  }
  float pcm_f32[kFrameSamples];
  for (int i = 0; i < kFrameSamples; ++i) {
    pcm_f32[i] = static_cast<float>(pcm_320[i]) / 32768.0f;
  }
  return kws_feed_pcm_f32(engine, pcm_f32);
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
  auto* impl = ToImpl(engine);
  if (impl->wake_hits >= kWakeHits) {
    return kSiriLabel;
  }
  const float* logits = impl->last_logits;
  int best = KWS_LABEL_SILENCE;
  float best_score = logits[KWS_LABEL_SILENCE];
  if (logits[KWS_LABEL_UNKNOWN] > best_score) {
    best = KWS_LABEL_UNKNOWN;
    best_score = logits[KWS_LABEL_UNKNOWN];
  }
  return best;
}

float kws_get_label_score(KwsEngine* engine, int label) {
  if (engine == nullptr || label < 0 || label >= kNumLabels) {
    return 0.f;
  }
  return ToImpl(engine)->last_logits[label];
}

float kws_get_label_prob(KwsEngine* engine, int label) {
  if (engine == nullptr || label < 0 || label >= kNumLabels) {
    return 0.f;
  }
  return ToImpl(engine)->last_probs[label];
}

int kws_is_wake(KwsEngine* engine) {
  if (engine == nullptr) {
    return 0;
  }
  return ToImpl(engine)->wake_hits >= kWakeHits ? 1 : 0;
}

}  // extern "C"
