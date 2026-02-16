// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#ifdef __CUDACC__
#define GTSAM_POINTS_HOST_DEVICE __host__ __device__
#else
#define GTSAM_POINTS_HOST_DEVICE
#endif

namespace gtsam_points {

/// @brief A pair of source and target point indices representing a correspondence.
struct Correspondence {
  GTSAM_POINTS_HOST_DEVICE Correspondence() : source_idx(-1), target_idx(-1) {}
  GTSAM_POINTS_HOST_DEVICE Correspondence(int source_idx, int target_idx) : source_idx(source_idx), target_idx(target_idx) {}

  int source_idx;
  int target_idx;
};

}  // namespace gtsam_points
