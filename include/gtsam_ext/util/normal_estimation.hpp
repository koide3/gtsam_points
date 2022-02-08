// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <vector>
#include <Eigen/Core>

namespace gtsam_ext {

std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
estimate_normals(const Eigen::Vector4d* points, const Eigen::Matrix4d* covs, int num_points, int k_neighbors = 10);

std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> estimate_normals(const Eigen::Vector4d* points, int num_points, int k_neighbors = 10);

template <template <typename> typename Alloc>
std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
estimate_normals(const std::vector<Eigen::Vector4d, Alloc<Eigen::Vector4d>>& points, const std::vector<Eigen::Matrix4d, Alloc<Eigen::Matrix4d>>& covs, int k_neighbors = 10) {
  return estimate_normals(points.data(), covs.data(), points.size(), k_neighbors);
}

template <typename Alloc>
std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> estimate_normals(
  const std::vector<Eigen::Vector4d, Alloc>& points,
  int k_neighbors = 10) {
  return estimate_normals(points.data(), points.size(), k_neighbors);
}

}  // namespace gtsam_ext