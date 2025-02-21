// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <vector>
#include <Eigen/Core>
#include <gtsam_points/types/point_cloud.hpp>

namespace gtsam_points {

/**
 * @brief Estimate point normals from covariances
 * @param points      Input points
 * @param covs        Input covariances
 * @param num_points  Number of input points and covariances
 * @param num_threads Number of threads
 * @return            Estimated normals
 */
std::vector<Eigen::Vector4d> estimate_normals(const Eigen::Vector4d* points, const Eigen::Matrix4d* covs, int num_points, int num_threads = 1);

/**
 * @brief Estimate point normals from neighboring points
 * @param points      Input points
 * @param num_points  Number of input points and covariances
 * @param k_neighbors Number of neighboring points for normal estimation
 * @param num_threads Number of threads
 * @return            Estimated normals
 */
std::vector<Eigen::Vector4d> estimate_normals(const Eigen::Vector4d* points, int num_points, int k_neighbors = 10, int num_threads = 1);

template <template <typename> typename Alloc>
std::vector<Eigen::Vector4d> estimate_normals(
  const std::vector<Eigen::Vector4d, Alloc<Eigen::Vector4d>>& points,
  const std::vector<Eigen::Matrix4d, Alloc<Eigen::Matrix4d>>& covs,
  int k_neighbors = 10,
  int num_threads = 1) {
  return estimate_normals(points.data(), covs.data(), points.size(), k_neighbors, num_threads);
}

template <typename Alloc>
std::vector<Eigen::Vector4d> estimate_normals(const std::vector<Eigen::Vector4d, Alloc>& points, int k_neighbors = 10, int num_threads = 1) {
  return estimate_normals(points.data(), points.size(), k_neighbors, num_threads);
}

std::vector<Eigen::Vector4d> estimate_normals(const PointCloud& points, int k_neighbors = 10, int num_threads = 1);

}  // namespace gtsam_points