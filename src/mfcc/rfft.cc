#include "mfcc/rfft.h"

#include <cstring>

#include "pffft.h"

namespace kws_mfcc {
namespace internal {

RealFft::~RealFft() {
  if (work_) pffft_aligned_free(work_);
  if (aligned_out_) pffft_aligned_free(aligned_out_);
  if (aligned_in_) pffft_aligned_free(aligned_in_);
  if (setup_) pffft_destroy_setup(setup_);
}

bool RealFft::Initialize(int n) {
  if (n < 32 || (n & (n - 1)) != 0) {
    return false;
  }
  setup_ = pffft_new_setup(n, PFFFT_REAL);
  if (setup_ == nullptr) {
    return false;
  }
  n_ = n;
  aligned_in_ = static_cast<float*>(pffft_aligned_malloc(
      static_cast<size_t>(n) * sizeof(float)));
  aligned_out_ = static_cast<float*>(pffft_aligned_malloc(
      static_cast<size_t>(n) * sizeof(float)));
  work_ = static_cast<float*>(pffft_aligned_malloc(
      static_cast<size_t>(n) * sizeof(float)));
  if (!aligned_in_ || !aligned_out_ || !work_) {
    return false;
  }
  return true;
}

void RealFft::Forward(const float* input, float* output) {
  std::memcpy(aligned_in_, input, static_cast<size_t>(n_) * sizeof(float));

  pffft_transform_ordered(setup_, aligned_in_, aligned_out_, work_,
                          PFFFT_FORWARD);

  // pffft ordered real output (length N):
  //   [0] = DC,  [1] = Nyquist,
  //   [2k] = Re(k),  [2k+1] = Im(k)  for k = 1..N/2-1
  //
  // Convert to TF rdft unpack layout (length N+2):
  //   [0] = DC_re,  [1] = 0,
  //   [2k] = Re(k), [2k+1] = Im(k)   for k = 1..N/2-1,
  //   [N] = Nyquist_re,  [N+1] = 0
  const int half = n_ / 2;

  output[0] = aligned_out_[0];
  output[1] = 0.0f;

  std::memcpy(output + 2, aligned_out_ + 2,
              static_cast<size_t>(half - 1) * 2 * sizeof(float));

  output[n_] = aligned_out_[1];
  output[n_ + 1] = 0.0f;
}

}  // namespace internal
}  // namespace kws_mfcc
