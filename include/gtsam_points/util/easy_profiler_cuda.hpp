// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>

struct CUevent_st;
struct CUstream_st;

namespace gtsam_points {
class EasyProfilerCuda {
public:
  EasyProfilerCuda(
    const std::string& prof_label,
    CUstream_st* stream,
    int events_cache_size = 10,
    bool enabled = true,
    bool debug = false,
    std::ostream& ost = std::cout);

  ~EasyProfilerCuda();

  void push(const std::string& label, CUstream_st* stream = nullptr);

private:
  CUevent_st* get_event();

public:
  const bool enabled;
  const bool debug;
  const std::string prof_label;

  std::vector<std::string> labels;
  std::vector<std::chrono::high_resolution_clock::time_point> times;
  std::vector<CUevent_st*> events;

  int events_count;
  std::vector<CUevent_st*> events_cache;

  std::ostream& ost;
  std::ofstream ofs;
};

}  // namespace gtsam_points
