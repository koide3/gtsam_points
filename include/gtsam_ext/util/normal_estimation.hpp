#pragma once

#include <vector>
#include <Eigen/Core>

namespace gtsam_ext {

std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> estimate_normals(
  const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
  const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs);

std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> estimate_normals(
  const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
  int k_neighbors = 10);
}  // namespace gtsam_ext