// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <vector>
#include <Eigen/Core>

namespace gtsam_ext {

/**
 * @brief Estimate point covariances from neighboring points
 * @param points      Input points
 * @param num_points  Number of input points
 * @param k_neighbors Number of neighboring points for covariance estimation
 * @param num_threads Number of threads
 * @return            Estimated covariances
 */
std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
estimate_covariances(const Eigen::Vector4d* points, int num_points, int k_neighbors = 10, int num_threads = 1);

template <typename Alloc>
std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
estimate_covariances(const std::vector<Eigen::Vector4d, Alloc>& points, int k_neighbors = 10, int num_threads = 1) {
  return estimate_covariances(points.data(), points.size(), k_neighbors, num_threads);
}

}  // namespace gtsam_ext