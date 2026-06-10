#include "hey_siri_kws.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <time.h>
#include <vector>

namespace {

double NowMs() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) * 1000.0 + static_cast<double>(ts.tv_nsec) / 1e6;
}

void FillTone(int16_t* pcm, int frame_idx) {
  for (int i = 0; i < KWS_FRAME_SAMPLES; ++i) {
    const float t = static_cast<float>(frame_idx * KWS_FRAME_SAMPLES + i) / 16000.0f;
    pcm[i] = static_cast<int16_t>(6000.0f * std::sin(2.0f * 3.14159265f * 440.0f * t));
  }
}

void PrintStats(const char* name, std::vector<double>& samples) {
  if (samples.empty()) {
    return;
  }
  std::sort(samples.begin(), samples.end());
  const size_t n = samples.size();
  double sum = 0.0;
  for (double v : samples) {
    sum += v;
  }
  const auto pct = [&](double p) {
    const size_t idx = static_cast<size_t>(p * static_cast<double>(n - 1));
    return samples[idx];
  };
  std::printf(
      "%s: n=%zu min=%.3f avg=%.3f p50=%.3f p90=%.3f p99=%.3f max=%.3f ms\n",
      name, n, samples.front(), sum / static_cast<double>(n), pct(0.50), pct(0.90),
      pct(0.99), samples.back());
}

long ReadRssKb() {
  FILE* fp = std::fopen("/proc/self/status", "r");
  if (fp == nullptr) {
    return -1;
  }
  char line[256];
  long rss = -1;
  while (std::fgets(line, sizeof(line), fp) != nullptr) {
    if (std::strncmp(line, "VmRSS:", 6) == 0) {
      std::sscanf(line + 6, "%ld", &rss);
      break;
    }
  }
  std::fclose(fp);
  return rss;
}

}  // namespace

int main() {
  constexpr int kWarmup = 50;
  constexpr int kIters = 500;

  std::printf("kws_bench_test start\n");
  std::printf("frame=%d samples (%.0f ms @16kHz), mfcc=%d\n",
              KWS_FRAME_SAMPLES, 1000.0 * KWS_FRAME_SAMPLES / 16000.0, KWS_MFCC_FEATURES);

  KwsEngine* engine = kws_create();
  if (engine == nullptr) {
    std::printf("FAIL: kws_create\n");
    return 1;
  }

  const long rss_after_create = ReadRssKb();
  std::printf("VmRSS after create: %ld KB\n", rss_after_create);

  int16_t pcm[KWS_FRAME_SAMPLES];
  float mfcc[KWS_MFCC_FEATURES];
  float logits[3];

  for (int i = 0; i < kWarmup; ++i) {
    FillTone(pcm, i);
    kws_feed_pcm_i16(engine, pcm);
  }

  std::vector<double> feed_ms;
  feed_ms.reserve(kIters);
  for (int i = 0; i < kIters; ++i) {
    FillTone(pcm, i + kWarmup);
    const double t0 = NowMs();
    if (kws_feed_pcm_i16(engine, pcm) != 0) {
      std::printf("FAIL: kws_feed_pcm_i16\n");
      kws_destroy(engine);
      return 1;
    }
    feed_ms.push_back(NowMs() - t0);
  }
  kws_get_logits(engine, logits);
  PrintStats("feed_pcm_i16 (mfcc+inference)", feed_ms);

  std::vector<double> mfcc_ms;
  mfcc_ms.reserve(kIters);
  for (int i = 0; i < kIters; ++i) {
    FillTone(pcm, i + kWarmup + kIters);
    const double t0 = NowMs();
    if (kws_compute_mfcc_i16(engine, pcm, mfcc) != 0) {
      std::printf("FAIL: kws_compute_mfcc_i16\n");
      kws_destroy(engine);
      return 1;
    }
    mfcc_ms.push_back(NowMs() - t0);
  }
  PrintStats("compute_mfcc_i16 only", mfcc_ms);

  const double avg_feed = [&]() {
    double s = 0.0;
    for (double v : feed_ms) {
      s += v;
    }
    return s / static_cast<double>(feed_ms.size());
  }();
  const double avg_mfcc = [&]() {
    double s = 0.0;
    for (double v : mfcc_ms) {
      s += v;
    }
    return s / static_cast<double>(mfcc_ms.size());
  }();
  const double est_infer = std::max(0.0, avg_feed - avg_mfcc);

  std::printf("estimated inference only: ~%.3f ms (feed - mfcc avg)\n", est_infer);
  std::printf("realtime factor (avg feed / 20ms frame): %.2fx\n", avg_feed / 20.0);
  std::printf("max sustainable fps at avg latency: %.1f\n", 1000.0 / avg_feed);

  kws_destroy(engine);
  std::printf("PASS: bench test\n");
  return 0;
}
