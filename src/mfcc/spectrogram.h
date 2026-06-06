#ifndef HEY_SIRI_KWS_SPECTROGRAM_H_
#define HEY_SIRI_KWS_SPECTROGRAM_H_

#include <complex>
#include <cstdint>
#include <deque>
#include <vector>

namespace kws_mfcc {
namespace internal {

class Spectrogram {
 public:
  Spectrogram();
  ~Spectrogram();

  bool Initialize(int window_length, int step_length);
  bool Initialize(const std::vector<double>& window, int step_length);

  template <typename InputSample, typename OutputSample>
  bool ComputeSquaredMagnitudeSpectrogram(
      const std::vector<InputSample>& input,
      std::vector<std::vector<OutputSample>>* output);

  int output_frequency_channels() const { return output_frequency_channels_; }

 private:
  template <typename InputSample>
  bool GetNextWindowOfSamples(const std::vector<InputSample>& input, int* input_start);

  void ProcessCoreFFT();

  int fft_length_;
  int output_frequency_channels_;
  int window_length_;
  int step_length_;
  bool initialized_;
  int samples_to_next_step_;

  std::vector<double> window_;
  std::vector<double> fft_input_output_;
  std::deque<double> input_queue_;

  std::vector<int> fft_integer_working_area_;
  std::vector<double> fft_double_working_area_;
};

}  // namespace internal
}  // namespace kws_mfcc

#endif  // HEY_SIRI_KWS_SPECTROGRAM_H_
