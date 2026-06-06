#ifndef HEY_SIRI_KWS_MFCC_DCT_H_
#define HEY_SIRI_KWS_MFCC_DCT_H_

#include <vector>

namespace kws_mfcc {
namespace internal {

class MfccDct {
 public:
  MfccDct();
  bool Initialize(int input_length, int coefficient_count);
  void Compute(const std::vector<double>& input, std::vector<double>* output) const;

 private:
  bool initialized_;
  int coefficient_count_;
  int input_length_;
  std::vector<std::vector<double>> cosines_;
};

}  // namespace internal
}  // namespace kws_mfcc

#endif  // HEY_SIRI_KWS_MFCC_DCT_H_
