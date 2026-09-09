#ifndef HEY_SIRI_KWS_RFFT_H_
#define HEY_SIRI_KWS_RFFT_H_

struct PFFFT_Setup;

namespace kws_mfcc {
namespace internal {

// Power-of-two real FFT backed by pffft (NEON on ARM, SSE on x86).
// Output layout matches TensorFlow rdft unpack:
//   output[0]=DC_re, output[1]=0,
//   output[2k]=Re(k), output[2k+1]=Im(k)  for k=1..N/2-1,
//   output[N]=Nyquist_re, output[N+1]=0.
class RealFft {
 public:
  RealFft() = default;
  ~RealFft();

  RealFft(const RealFft&) = delete;
  RealFft& operator=(const RealFft&) = delete;

  bool Initialize(int n);
  void Forward(const float* input, float* output);

  int size() const { return n_; }

 private:
  int n_ = 0;
  PFFFT_Setup* setup_ = nullptr;
  float* aligned_in_ = nullptr;
  float* aligned_out_ = nullptr;
  float* work_ = nullptr;
};

}  // namespace internal
}  // namespace kws_mfcc

#endif  // HEY_SIRI_KWS_RFFT_H_
