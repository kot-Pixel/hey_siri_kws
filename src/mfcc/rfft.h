#ifndef HEY_SIRI_KWS_RFFT_H_
#define HEY_SIRI_KWS_RFFT_H_

#include <vector>

namespace kws_mfcc {
namespace internal {

// Power-of-two real FFT. Output layout matches TensorFlow rdft unpack.
class RealFft {
 public:
  bool Initialize(int n);
  void Forward(const float* input, float* output);

  int size() const { return n_; }

 private:
  int n_ = 0;
  int log2_n_ = 0;
  std::vector<float> twiddle_cos_;
  std::vector<float> twiddle_sin_;
  std::vector<float> work_real_;
  std::vector<float> work_imag_;
};

}  // namespace internal
}  // namespace kws_mfcc

#endif  // HEY_SIRI_KWS_RFFT_H_
