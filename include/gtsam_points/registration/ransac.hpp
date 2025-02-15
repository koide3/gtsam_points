// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#pragma once

#include <random>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam_points/ann/nearest_neighbor_search.hpp>
#include <gtsam_points/registration/registration_result.hpp>

namespace gtsam_points {

template <typename PointCloud, typename Features>
RegistrationResult estimate_pose_ransac(
  const PointCloud& target,
  const PointCloud& source,
  const Features& target_features,
  const Features& source_features,
  const NearestNeighborSearch& target_tree,
  const NearestNeighborSearch& target_features_tree,
  double inlier_voxel_resolution,
  int max_iterations,
  std::mt19937& mt);

}  // namespace gtsam_points
