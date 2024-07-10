// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/util/easy_profiler_cuda.hpp>

#include <cuda_runtime.h>
#include <gtsam_points/cuda/check_error.cuh>

namespace gtsam_points {

EasyProfilerCuda::EasyProfilerCuda(
  const std::string& prof_label,
  CUstream_st* stream,
  int event_cache_size,
  bool enabled,
  bool debug,
  std::ostream& ost)
: enabled(enabled),
  debug(debug),
  prof_label(prof_label),
  ost(ost) {
  if (!enabled) {
    return;
  }

  events_count = 0;
  events_cache.resize(event_cache_size, nullptr);
  for (int i = 0; i < event_cache_size; i++) {
    cudaEvent_t event;
    check_error << cudaEventCreate(&event);
    events_cache[i] = event;
  }

  push("begin", stream);
}

EasyProfilerCuda::~EasyProfilerCuda() {
  if (!enabled) {
    return;
  }

  push("end");
  check_error << cudaEventSynchronize(events.back());

  ost << "--- " << prof_label << " ---\n";

  int longest = 0;
  for (const auto& label : labels) {
    longest = std::max<int>(label.size(), longest);
  }

  for (int i = 1; i < labels.size(); i++) {
    std::vector<char> pad(longest - labels[i - 1].size(), ' ');
    std::string label = labels[i - 1] + std::string(pad.begin(), pad.end());

    double msec_cpu = std::chrono::duration_cast<std::chrono::nanoseconds>(times[i] - times[i - 1]).count() / 1e6;
    double msec_cpu_accum = std::chrono::duration_cast<std::chrono::nanoseconds>(times[i] - times.front()).count() / 1e6;

    float msec_gpu;
    float msec_gpu_accum;
    check_error << cudaEventElapsedTime(&msec_gpu, events[i - 1], events[i]);
    check_error << cudaEventElapsedTime(&msec_gpu_accum, events.front(), events[i]);

    ost << label << ": CPU=" << msec_cpu << "[msec] (accum=" << msec_cpu_accum << "[msec]) GPU=" << msec_gpu << "[msec] (accum=" << msec_gpu_accum
        << "[msec])" << '\n';
  }

  ost << "total:" << std::chrono::duration_cast<std::chrono::nanoseconds>(times.back() - times.front()).count() / 1e6 << "[msec]" << '\n';
  ost << std::flush;

  for (int i = 0; i < events_cache.size(); i++) {
    check_error << cudaEventDestroy(events_cache[i]);
  }
}

void EasyProfilerCuda::push(const std::string& label, CUstream_st* stream) {
  if (!enabled) {
    return;
  }

  labels.emplace_back(label);
  times.emplace_back(std::chrono::high_resolution_clock::now());
  events.emplace_back(get_event());
  check_error << cudaEventRecord(events.back(), stream);

  if (debug) {
    double msec_cpu = std::chrono::duration_cast<std::chrono::nanoseconds>(times.back() - times.front()).count() / 1e6;

    check_error << cudaEventSynchronize(events.back());
    float msec_gpu = 0.0f;
    check_error << cudaEventElapsedTime(&msec_gpu, events.front(), events.back());
    ost << ">> " << label << " (CPU=" << msec_cpu << "[msec] GPU=" << msec_gpu << "[msec])" << std::endl;
  }
}

CUevent_st* EasyProfilerCuda::get_event() {
  while (events_count >= events_cache.size()) {
    cudaEvent_t event;
    check_error << cudaEventCreate(&event);
    events_cache.emplace_back(event);
  }

  return events_cache[events_count++];
}

}  // namespace gtsam_points
