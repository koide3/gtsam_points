// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <chrono>

namespace gtsam_points {

/// @brief Just a stopwatch
struct Stopwatch {
public:
  void start() { t1 = t2 = std::chrono::high_resolution_clock::now(); }
  void stop() { t2 = std::chrono::high_resolution_clock::now(); }
  void lap() {
    t1 = t2;
    t2 = std::chrono::high_resolution_clock::now();
  }

  double sec() const { return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e9; }
  double msec() const { return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6; }

public:
  std::chrono::high_resolution_clock::time_point t1;
  std::chrono::high_resolution_clock::time_point t2;
};

}  // namespace gtsam_points
