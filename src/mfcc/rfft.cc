#include "mfcc/rfft.h"

#include <cmath>

namespace kws_mfcc {
namespace internal {

namespace {

int Log2(int n) {
  int log = 0;
  while ((1 << log) < n) {
    ++log;
  }
  return log;
}

void BitReversePermute(std::vector<float>* real, std::vector<float>* imag, int n) {
  int j = 0;
  for (int i = 1; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      std::swap((*real)[i], (*real)[j]);
      std::swap((*imag)[i], (*imag)[j]);
    }
  }
}

}  // namespace

bool RealFft::Initialize(int n) {
  if (n < 2 || (n & (n - 1)) != 0) {
    return false;
  }
  n_ = n;
  log2_n_ = Log2(n);
  const int half = n / 2;
  twiddle_cos_.resize(half);
  twiddle_sin_.resize(half);
  const float pi = std::acos(-1.0f);
  for (int i = 0; i < half; ++i) {
    const float angle = -2.0f * pi * static_cast<float>(i) / static_cast<float>(n);
    twiddle_cos_[i] = std::cos(angle);
    twiddle_sin_[i] = std::sin(angle);
  }
  work_real_.assign(n, 0.0f);
  work_imag_.assign(n, 0.0f);
  return true;
}

void RealFft::Forward(const float* input, float* output) {
  work_real_.assign(n_, 0.0f);
  work_imag_.assign(n_, 0.0f);
  for (int i = 0; i < n_; ++i) {
    work_real_[i] = input[i];
  }

  BitReversePermute(&work_real_, &work_imag_, n_);

  for (int len = 2; len <= n_; len <<= 1) {
    const int half_len = len / 2;
    const int step = n_ / len;
    for (int i = 0; i < n_; i += len) {
      for (int j = 0; j < half_len; ++j) {
        const int tw_idx = j * step;
        const float cos_v = twiddle_cos_[tw_idx];
        const float sin_v = twiddle_sin_[tw_idx];
        const int even = i + j;
        const int odd = i + j + half_len;
        const float t_real = work_real_[odd] * cos_v - work_imag_[odd] * sin_v;
        const float t_imag = work_real_[odd] * sin_v + work_imag_[odd] * cos_v;
        work_real_[odd] = work_real_[even] - t_real;
        work_imag_[odd] = work_imag_[even] - t_imag;
        work_real_[even] += t_real;
        work_imag_[even] += t_imag;
      }
    }
  }

  const int out_bins = n_ / 2 + 1;
  for (int k = 0; k < out_bins; ++k) {
    output[2 * k] = work_real_[k];
    output[2 * k + 1] = work_imag_[k];
  }
  output[1] = 0.0f;
  output[n_] = work_real_[n_ / 2];
  output[n_ + 1] = 0.0f;
}

}  // namespace internal
}  // namespace kws_mfcc
