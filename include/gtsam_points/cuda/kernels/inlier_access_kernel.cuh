// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <iostream>

namespace gtsam_points {

template <bool enable_pretrimming = true>
struct inlier_access_kernel {
  inlier_access_kernel(const int* source_inliers) : source_inliers(source_inliers) { std::cout << "enabled" << std::endl; }

  __host__ __device__ int operator()(int i) const { return source_inliers[i]; }

  const int* source_inliers;
};

template <>
struct inlier_access_kernel<false> {
  inlier_access_kernel(const int* source_inliers) : source_inliers(source_inliers) {}

  __host__ __device__ int operator()(int i) const { return i; }

  const int* source_inliers;
};

}  // namespace gtsam_points