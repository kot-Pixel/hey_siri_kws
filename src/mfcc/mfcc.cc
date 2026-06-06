#include "mfcc/mfcc.h"

#include <cmath>

namespace kws_mfcc {
namespace internal {

namespace {
constexpr double kDefaultUpperFrequencyLimit = 4000.0;
constexpr double kDefaultLowerFrequencyLimit = 20.0;
constexpr double kFilterbankFloor = 1e-12;
constexpr int kDefaultFilterbankChannelCount = 40;
constexpr int kDefaultDctCoefficientCount = 13;
}  // namespace

Mfcc::Mfcc()
    : initialized_(false),
      lower_frequency_limit_(kDefaultLowerFrequencyLimit),
      upper_frequency_limit_(kDefaultUpperFrequencyLimit),
      filterbank_channel_count_(kDefaultFilterbankChannelCount),
      dct_coefficient_count_(kDefaultDctCoefficientCount) {}

bool Mfcc::Initialize(int input_length, double input_sample_rate) {
  bool initialized = mel_filterbank_.Initialize(
      input_length, input_sample_rate, filterbank_channel_count_, lower_frequency_limit_,
      upper_frequency_limit_);
  initialized &= dct_.Initialize(filterbank_channel_count_, dct_coefficient_count_);
  initialized_ = initialized;
  return initialized;
}

void Mfcc::Compute(const std::vector<double>& spectrogram_frame,
                   std::vector<double>* output) const {
  if (!initialized_) {
    return;
  }

  std::vector<double> working;
  mel_filterbank_.Compute(spectrogram_frame, &working);
  for (double& val : working) {
    if (val < kFilterbankFloor) {
      val = kFilterbankFloor;
    }
    val = std::log(val);
  }
  dct_.Compute(working, output);
}

}  // namespace internal
}  // namespace kws_mfcc
