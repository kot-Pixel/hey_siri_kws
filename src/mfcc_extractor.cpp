#include "mfcc_extractor.h"

#include <cmath>
#include <vector>

#include "mfcc/mfcc.h"
#include "mfcc/spectrogram.h"

namespace kws_mfcc {

class MfccExtractor::Impl {
 public:
  internal::Spectrogram spectrogram;
  internal::Mfcc mfcc;
  int spectrogram_channels = 0;
};

MfccExtractor::MfccExtractor(const MfccConfig& config)
    : config_(config), initialized_(false), impl_(std::make_unique<Impl>()) {}

MfccExtractor::~MfccExtractor() = default;

bool MfccExtractor::Initialize() {
  if (config_.window_size_samples < 2 || config_.window_stride_samples < 1 ||
      config_.mel_num_bins < 1 || config_.dct_num_features < 1) {
    return false;
  }

  if (!impl_->spectrogram.Initialize(config_.window_size_samples,
                                     config_.window_stride_samples)) {
    return false;
  }

  impl_->spectrogram_channels = impl_->spectrogram.output_frequency_channels();
  impl_->mfcc.set_upper_frequency_limit(config_.mel_upper_edge_hertz);
  impl_->mfcc.set_lower_frequency_limit(config_.mel_lower_edge_hertz);
  impl_->mfcc.set_filterbank_channel_count(config_.mel_num_bins);
  impl_->mfcc.set_dct_coefficient_count(config_.dct_num_features);
  if (!impl_->mfcc.Initialize(impl_->spectrogram_channels, config_.sample_rate)) {
    return false;
  }

  initialized_ = true;
  return true;
}

void MfccExtractor::Reset() {
  initialized_ = false;
  impl_ = std::make_unique<Impl>();
  Initialize();
}

bool MfccExtractor::ComputeFrame(const float* pcm_frame, float* mfcc_out) {
  if (!initialized_ || pcm_frame == nullptr || mfcc_out == nullptr) {
    return false;
  }

  std::vector<float> pcm(config_.window_size_samples);
  for (int i = 0; i < config_.window_size_samples; ++i) {
    pcm[i] = pcm_frame[i];
  }

  std::vector<std::vector<float>> squared_spectrogram;
  if (!impl_->spectrogram.ComputeSquaredMagnitudeSpectrogram(pcm, &squared_spectrogram) ||
      squared_spectrogram.empty()) {
    return false;
  }

  std::vector<double> spectrum(impl_->spectrogram_channels);
  for (int i = 0; i < impl_->spectrogram_channels; ++i) {
    double value = squared_spectrogram[0][i];
    if (!config_.magnitude_squared) {
      value = std::sqrt(value);
    }
    spectrum[i] = value;
  }

  std::vector<double> mfcc;
  impl_->mfcc.Compute(spectrum, &mfcc);
  if (static_cast<int>(mfcc.size()) != config_.dct_num_features) {
    return false;
  }

  for (int i = 0; i < config_.dct_num_features; ++i) {
    mfcc_out[i] = static_cast<float>(mfcc[i]);
  }
  return true;
}

}  // namespace kws_mfcc
