#ifndef HEY_SIRI_KWS_MFCC_H_
#define HEY_SIRI_KWS_MFCC_H_

#include <vector>

#include "mfcc/mfcc_dct.h"
#include "mfcc/mfcc_mel_filterbank.h"

namespace kws_mfcc {
namespace internal {

class Mfcc {
 public:
  Mfcc();

  bool Initialize(int input_length, double input_sample_rate);
  void Compute(const std::vector<double>& spectrogram_frame,
               std::vector<double>* output) const;

  void set_upper_frequency_limit(double upper_frequency_limit) {
    upper_frequency_limit_ = upper_frequency_limit;
  }

  void set_lower_frequency_limit(double lower_frequency_limit) {
    lower_frequency_limit_ = lower_frequency_limit;
  }

  void set_filterbank_channel_count(int filterbank_channel_count) {
    filterbank_channel_count_ = filterbank_channel_count;
  }

  void set_dct_coefficient_count(int dct_coefficient_count) {
    dct_coefficient_count_ = dct_coefficient_count;
  }

 private:
  MfccMelFilterbank mel_filterbank_;
  MfccDct dct_;
  bool initialized_;
  double lower_frequency_limit_;
  double upper_frequency_limit_;
  int filterbank_channel_count_;
  int dct_coefficient_count_;
};

}  // namespace internal
}  // namespace kws_mfcc

#endif  // HEY_SIRI_KWS_MFCC_H_
