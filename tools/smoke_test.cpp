#include "hey_siri_kws.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace {

bool IsFinite3(const float* v) {
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(v[i])) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  std::printf("kws_smoke_test start\n");

  KwsEngine* engine = kws_create();
  if (engine == nullptr) {
    std::printf("FAIL: kws_create returned null\n");
    return 1;
  }
  std::printf("OK: kws_create\n");

  int16_t silence[KWS_FRAME_SAMPLES] = {};
  float logits[3] = {};

  for (int frame = 0; frame < 100; ++frame) {
    if (kws_feed_pcm_i16(engine, silence) != 0) {
      std::printf("FAIL: kws_feed_pcm_i16 silence frame %d\n", frame);
      kws_destroy(engine);
      return 1;
    }
  }
  if (kws_get_logits(engine, logits) != 0 || !IsFinite3(logits)) {
    std::printf("FAIL: invalid logits after silence\n");
    kws_destroy(engine);
    return 1;
  }
  std::printf("OK: 100 silence frames, top=%d hey=%.4f\n",
              kws_get_top_label(engine), kws_get_label_score(engine, KWS_LABEL_HEY_SIRI));

  // Simple tone burst (not a real wake word; just checks inference path).
  int16_t tone[KWS_FRAME_SAMPLES];
  for (int frame = 0; frame < 50; ++frame) {
    for (int i = 0; i < KWS_FRAME_SAMPLES; ++i) {
      const float t = static_cast<float>(frame * KWS_FRAME_SAMPLES + i) / 16000.0f;
      tone[i] = static_cast<int16_t>(8000.0f * std::sin(2.0f * 3.14159265f * 440.0f * t));
    }
    if (kws_feed_pcm_i16(engine, tone) != 0) {
      std::printf("FAIL: kws_feed_pcm_i16 tone frame %d\n", frame);
      kws_destroy(engine);
      return 1;
    }
  }
  if (kws_get_logits(engine, logits) != 0 || !IsFinite3(logits)) {
    std::printf("FAIL: invalid logits after tone\n");
    kws_destroy(engine);
    return 1;
  }
  std::printf("OK: 50 tone frames, top=%d hey=%.4f\n",
              kws_get_top_label(engine), kws_get_label_score(engine, KWS_LABEL_HEY_SIRI));

  float mfcc[KWS_MFCC_FEATURES];
  if (kws_compute_mfcc_i16(engine, silence, mfcc) != 0) {
    std::printf("FAIL: kws_compute_mfcc_i16\n");
    kws_destroy(engine);
    return 1;
  }
  for (int i = 0; i < KWS_MFCC_FEATURES; ++i) {
    if (!std::isfinite(mfcc[i])) {
      std::printf("FAIL: non-finite mfcc[%d]\n", i);
      kws_destroy(engine);
      return 1;
    }
  }
  std::printf("OK: mfcc extract (first coeff=%.4f)\n", mfcc[0]);

  kws_reset_mfcc(engine);
  kws_destroy(engine);
  std::printf("OK: kws_destroy\n");
  std::printf("PASS: smoke test\n");
  return 0;
}
