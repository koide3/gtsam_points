#pragma once

#include <vector>
#include <Eigen/Core>

namespace gtsam_ext {

std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> estimate_covariances(
  const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
  int k_neighbors = 10);

}