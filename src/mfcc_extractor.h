#ifndef HEY_SIRI_KWS_MFCC_EXTRACTOR_H_
#define HEY_SIRI_KWS_MFCC_EXTRACTOR_H_

#include <memory>

namespace kws_mfcc {

struct MfccConfig {
  int sample_rate = 16000;
  int window_size_samples = 320;
  int window_stride_samples = 320;
  int mel_num_bins = 40;
  int dct_num_features = 20;
  float mel_lower_edge_hertz = 20.0f;
  float mel_upper_edge_hertz = 7600.0f;
  bool magnitude_squared = false;
};

// Streaming MFCC extractor matching kws_streaming preprocess=mfcc
// (TensorFlow audio_ops.audio_spectrogram + audio_ops.mfcc).
class MfccExtractor {
 public:
  static constexpr int kFrameSamples = 320;
  static constexpr int kNumFeatures = 20;

  explicit MfccExtractor(const MfccConfig& config = MfccConfig());
  ~MfccExtractor();

  MfccExtractor(const MfccExtractor&) = delete;
  MfccExtractor& operator=(const MfccExtractor&) = delete;

  bool Initialize();
  void Reset();

  // PCM in [-1, 1]. Writes dct_num_features MFCC coefficients on success.
  bool ComputeFrame(const float* pcm_frame, float* mfcc_out);

  int num_features() const { return config_.dct_num_features; }

 private:
  MfccConfig config_;
  bool initialized_;
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kws_mfcc

#endif  // HEY_SIRI_KWS_MFCC_EXTRACTOR_H_
