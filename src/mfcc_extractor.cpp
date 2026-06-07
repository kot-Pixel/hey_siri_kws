#include "mfcc_extractor.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "mfcc/rfft.h"

namespace kws_mfcc {

namespace {
constexpr float kFilterbankFloor = 1e-12f;
constexpr float kPi = 3.14159265358979323846f;

float FreqToMel(float freq) {
  return 1127.0f * std::log1p(freq / 700.0f);
}

int NextPowerOfTwo(int value) {
  int n = 1;
  while (n < value) {
    n <<= 1;
  }
  return n;
}

}  // namespace

class MfccExtractor::Impl {
 public:
  internal::RealFft fft;
  int fft_length = 0;
  int spectrum_bins = 0;
  int mel_start = 0;
  int mel_end = 0;

  std::vector<float> hann_window;
  std::vector<float> fft_out;
  std::vector<float> spectrum;
  std::vector<float> mel_energies;
  std::vector<float> mel_weights;
  std::vector<int16_t> mel_band_mapper;
  std::vector<float> dct_matrix;
  std::vector<float> windowed;

  bool BuildMelFilterbank(const MfccConfig& config) {
    const int input_length = spectrum_bins;
    const int num_channels = config.mel_num_bins;
    const float sample_rate = static_cast<float>(config.sample_rate);
    const float lower = config.mel_lower_edge_hertz;
    const float upper = config.mel_upper_edge_hertz;

    std::vector<float> center_frequencies(num_channels + 1);
    const float mel_low = FreqToMel(lower);
    const float mel_hi = FreqToMel(upper);
    const float mel_spacing = (mel_hi - mel_low) / static_cast<float>(num_channels + 1);
    for (int i = 0; i < num_channels + 1; ++i) {
      center_frequencies[i] = mel_low + mel_spacing * static_cast<float>(i + 1);
    }

    const float hz_per_sbin = 0.5f * sample_rate / static_cast<float>(input_length - 1);
    mel_start = static_cast<int>(1.5f + (lower / hz_per_sbin));
    mel_end = static_cast<int>(upper / hz_per_sbin);
    mel_start = std::max(mel_start, 1);
    mel_end = std::min(mel_end, input_length - 1);

    std::vector<int> band_mapper(input_length, -2);
    int channel = 0;
    for (int i = 0; i < input_length; ++i) {
      const float melf = FreqToMel(static_cast<float>(i) * hz_per_sbin);
      if (i < mel_start || i > mel_end) {
        band_mapper[i] = -2;
      } else {
        while (channel < num_channels && center_frequencies[channel] < melf) {
          ++channel;
        }
        band_mapper[i] = channel - 1;
      }
    }

    std::vector<float> weights(input_length, 0.0f);
    for (int i = 0; i < input_length; ++i) {
      channel = band_mapper[i];
      if (i < mel_start || i > mel_end) {
        weights[i] = 0.0f;
      } else if (channel >= 0) {
        weights[i] = (center_frequencies[channel + 1] - FreqToMel(static_cast<float>(i) * hz_per_sbin)) /
                     (center_frequencies[channel + 1] - center_frequencies[channel]);
      } else {
        weights[i] = (center_frequencies[0] - FreqToMel(static_cast<float>(i) * hz_per_sbin)) /
                       (center_frequencies[0] - mel_low);
      }
    }

    mel_weights = weights;
    mel_band_mapper.resize(input_length);
    for (int i = 0; i < input_length; ++i) {
      mel_band_mapper[i] = static_cast<int16_t>(band_mapper[i]);
    }
    return true;
  }

  bool BuildDctMatrix(int num_mel_bins, int num_features) {
    dct_matrix.resize(static_cast<size_t>(num_features) * num_mel_bins);
    const float fnorm = std::sqrt(2.0f / static_cast<float>(num_mel_bins));
    const float arg = kPi / static_cast<float>(num_mel_bins);
    for (int i = 0; i < num_features; ++i) {
      for (int j = 0; j < num_mel_bins; ++j) {
        dct_matrix[static_cast<size_t>(i) * num_mel_bins + j] =
            fnorm * std::cos(static_cast<float>(i) * arg * (static_cast<float>(j) + 0.5f));
      }
    }
    return true;
  }
};

MfccExtractor::MfccExtractor(const MfccConfig& config)
    : config_(config), initialized_(false), impl_(std::make_unique<Impl>()) {}

MfccExtractor::~MfccExtractor() = default;

bool MfccExtractor::Initialize() {
  if (config_.window_size_samples < 2 || config_.window_stride_samples < 1 ||
      config_.mel_num_bins < 1 || config_.dct_num_features < 1) {
    return false;
  }

  impl_->fft_length = NextPowerOfTwo(config_.window_size_samples);
  impl_->spectrum_bins = 1 + impl_->fft_length / 2;
  if (!impl_->fft.Initialize(impl_->fft_length)) {
    return false;
  }

  impl_->hann_window.resize(config_.window_size_samples);
  for (int i = 0; i < config_.window_size_samples; ++i) {
    impl_->hann_window[i] =
        0.5f - 0.5f * std::cos(2.0f * kPi * static_cast<float>(i) /
                               static_cast<float>(config_.window_size_samples));
  }

  impl_->fft_out.assign(impl_->fft_length + 2, 0.0f);
  impl_->spectrum.assign(impl_->spectrum_bins, 0.0f);
  impl_->mel_energies.assign(config_.mel_num_bins, 0.0f);
  impl_->windowed.assign(impl_->fft_length, 0.0f);

  if (!impl_->BuildMelFilterbank(config_) || !impl_->BuildDctMatrix(config_.mel_num_bins, config_.dct_num_features)) {
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

  std::fill(impl_->windowed.begin(), impl_->windowed.end(), 0.0f);
  for (int i = 0; i < config_.window_size_samples; ++i) {
    impl_->windowed[i] = pcm_frame[i] * impl_->hann_window[i];
  }

  impl_->fft.Forward(impl_->windowed.data(), impl_->fft_out.data());

  for (int i = 0; i < impl_->spectrum_bins; ++i) {
    const float re = impl_->fft_out[2 * i];
    const float im = impl_->fft_out[2 * i + 1];
    float power = re * re + im * im;
    if (!config_.magnitude_squared) {
      power = std::sqrt(power);
    }
    impl_->spectrum[i] = power;
  }

  std::fill(impl_->mel_energies.begin(), impl_->mel_energies.end(), 0.0f);
  for (int i = impl_->mel_start; i <= impl_->mel_end; ++i) {
    const float spec_val = std::sqrt(impl_->spectrum[i]);
    const float weighted = spec_val * impl_->mel_weights[i];
    int channel = impl_->mel_band_mapper[i];
    if (channel >= 0) {
      impl_->mel_energies[channel] += weighted;
    }
    ++channel;
    if (channel < config_.mel_num_bins) {
      impl_->mel_energies[channel] += spec_val - weighted;
    }
  }

  for (int i = 0; i < config_.mel_num_bins; ++i) {
    float val = impl_->mel_energies[i];
    if (val < kFilterbankFloor) {
      val = kFilterbankFloor;
    }
    impl_->mel_energies[i] = std::log(val);
  }

  for (int i = 0; i < config_.dct_num_features; ++i) {
    float sum = 0.0f;
    const float* row = impl_->dct_matrix.data() + static_cast<size_t>(i) * config_.mel_num_bins;
    for (int j = 0; j < config_.mel_num_bins; ++j) {
      sum += row[j] * impl_->mel_energies[j];
    }
    mfcc_out[i] = sum;
  }

  return true;
}

}  // namespace kws_mfcc
