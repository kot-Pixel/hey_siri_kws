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
};

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

#ifdef __cplusplus
}
#endif

#endif  // HEY_SIRI_KWS_H_
