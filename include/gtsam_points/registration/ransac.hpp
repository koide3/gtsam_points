// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#pragma once

#include <random>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam_points/ann/nearest_neighbor_search.hpp>
#include <gtsam_points/registration/registration_result.hpp>

namespace gtsam_points {

/// @brief RANSAC parameters
struct RANSACParams {
  int max_iterations = 5000;             ///< Maximum number of iterations
  double early_stop_inlier_rate = 0.8;   ///< Maximum inlier rate for early stopping
  double poly_error_thresh = 0.5;        ///< Polynomial error threshold
  double inlier_voxel_resolution = 1.0;  ///< Inlier voxel resolution
  int dof = 6;                           ///< Degrees of freedom (must be 6 (SE3) or 4 (XYZ+RZ))
  std::uint64_t seed = 5489u;            ///< Random seed
  int num_threads = 4;                   ///< Number of threads
};

template <typename PointCloud, typename Features>
RegistrationResult estimate_pose_ransac(
  const PointCloud& target,
  const PointCloud& source,
  const Features& target_features,
  const Features& source_features,
  const NearestNeighborSearch& target_tree,
  const NearestNeighborSearch& target_features_tree,
  const RANSACParams& params);

}  // namespace gtsam_points
