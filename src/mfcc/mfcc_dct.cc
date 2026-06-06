#include "mfcc/mfcc_dct.h"

#include <cmath>

namespace kws_mfcc {
namespace internal {

MfccDct::MfccDct() : initialized_(false) {}

bool MfccDct::Initialize(int input_length, int coefficient_count) {
  coefficient_count_ = coefficient_count;
  input_length_ = input_length;

  if (coefficient_count_ < 1 || input_length < 1 ||
      coefficient_count_ > input_length_) {
    return false;
  }

  cosines_.resize(coefficient_count_);
  const double fnorm = std::sqrt(2.0 / input_length_);
  const double pi = std::atan(1.0) * 4.0;
  const double arg = pi / input_length_;
  for (int i = 0; i < coefficient_count_; ++i) {
    cosines_[i].resize(input_length_);
    for (int j = 0; j < input_length_; ++j) {
      cosines_[i][j] = fnorm * std::cos(i * arg * (j + 0.5));
    }
  }
  initialized_ = true;
  return true;
}

void MfccDct::Compute(const std::vector<double>& input,
                      std::vector<double>* output) const {
  if (!initialized_) {
    return;
  }

  output->resize(coefficient_count_);
  int length = static_cast<int>(input.size());
  if (length > input_length_) {
    length = input_length_;
  }

  for (int i = 0; i < coefficient_count_; ++i) {
    double sum = 0.0;
    for (int j = 0; j < length; ++j) {
      sum += cosines_[i][j] * input[j];
    }
    (*output)[i] = sum;
  }
}

}  // namespace internal
}  // namespace kws_mfcc
