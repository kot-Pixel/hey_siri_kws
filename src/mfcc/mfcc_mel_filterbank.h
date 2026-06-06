#ifndef HEY_SIRI_KWS_MFCC_MEL_FILTERBANK_H_
#define HEY_SIRI_KWS_MFCC_MEL_FILTERBANK_H_

#include <vector>

namespace kws_mfcc {
namespace internal {

class MfccMelFilterbank {
 public:
  MfccMelFilterbank();
  bool Initialize(int input_length, double input_sample_rate, int output_channel_count,
                  double lower_frequency_limit, double upper_frequency_limit);
  void Compute(const std::vector<double>& input, std::vector<double>* output) const;

 private:
  double FreqToMel(double freq) const;

  bool initialized_;
  int num_channels_;
  double sample_rate_;
  int input_length_;
  std::vector<double> center_frequencies_;
  std::vector<double> weights_;
  std::vector<int> band_mapper_;
  int start_index_;
  int end_index_;
};

}  // namespace internal
}  // namespace kws_mfcc

#endif  // HEY_SIRI_KWS_MFCC_MEL_FILTERBANK_H_
