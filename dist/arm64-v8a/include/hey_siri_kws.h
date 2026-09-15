#ifndef HEY_SIRI_KWS_H_
#define HEY_SIRI_KWS_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum KwsLabel {
  KWS_LABEL_SILENCE = 0,
  KWS_LABEL_UNKNOWN = 1,
  KWS_LABEL_HEY_SIRI = 2,
};

enum {
  KWS_FRAME_SAMPLES = 320,
  KWS_MFCC_FEATURES = 20,
  KWS_MFCC_FRAMES = 50,
  // Clip model is ~7.3M ops. Run it every N 20ms frames (80ms) instead of
  // every frame; the 1s MFCC window still shifts every 20ms.
  KWS_INFER_STRIDE = 4,
  // Consecutive inferences above KWS_WAKE_THRESHOLD before top_label is siri.
  KWS_WAKE_HITS = 2,
};

// Softmax(siri) must stay at or above this for KWS_WAKE_HITS inferences.
#define KWS_WAKE_THRESHOLD 0.90f

typedef struct KwsEngine KwsEngine;

KwsEngine* kws_create(void);
void kws_destroy(KwsEngine* engine);

int kws_feed_pcm_f32(KwsEngine* engine, const float* pcm_320);
int kws_feed_pcm_i16(KwsEngine* engine, const int16_t* pcm_320);

// PCM -> MFCC (matches model_mfcc/flags.json). Also updated on each kws_feed_pcm_*.
int kws_compute_mfcc_f32(KwsEngine* engine, const float* pcm_320, float mfcc_20[KWS_MFCC_FEATURES]);
int kws_compute_mfcc_i16(KwsEngine* engine, const int16_t* pcm_320, float mfcc_20[KWS_MFCC_FEATURES]);
int kws_get_last_mfcc(KwsEngine* engine, float mfcc_20[KWS_MFCC_FEATURES]);
void kws_reset_mfcc(KwsEngine* engine);

int kws_get_logits(KwsEngine* engine, float out_logits[3]);
int kws_get_top_label(KwsEngine* engine);
float kws_get_label_score(KwsEngine* engine, int label);
float kws_get_label_prob(KwsEngine* engine, int label);
int kws_is_wake(KwsEngine* engine);

#ifdef __cplusplus
}
#endif

#endif  // HEY_SIRI_KWS_H_
