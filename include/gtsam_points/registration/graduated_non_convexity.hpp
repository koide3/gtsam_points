// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#pragma once

#include <random>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam_points/ann/nearest_neighbor_search.hpp>
#include <gtsam_points/registration/registration_result.hpp>

namespace gtsam_points {

struct GNCParams {
public:
  // Feature matching parameters
  int max_init_samples = 5000;   ///< Maximum number of samples
  bool reciprocal_check = true;  ///< Reciprocal check
  double tuple_thresh = 0.9;     ///< Length similarity threshold
  int max_num_tuples = 1000;     ///< Number of tuples to be sampled

  // Estimation praameters
  double div_factor = 1.4;     ///< Division factor for graduated non-convexity
  double max_corr_dist = 0.1;  ///< Maximum correspondence distance
  int innter_iterations = 5;   ///< Number of inner iterations
  int max_iterations = 64;     ///< Maximum number of iterations
  int dof = 6;                 ///< Degrees of freedom (must be 6 (SE3) or 4 (XYZ+RZ))

  // Misc
  bool verbose = false;        ///< Verbose mode
  std::uint64_t seed = 5489u;  ///< Random seed
  int num_threads = 4;         ///< Number of threads
};

template <typename PointCloud, typename Features>
RegistrationResult estimate_pose_gnc(
  const PointCloud& target,
  const PointCloud& source,
  const Features& target_features,
  const Features& source_features,
  const NearestNeighborSearch& target_tree,
  const NearestNeighborSearch& target_features_tree,
  const NearestNeighborSearch& source_features_tree,
  const GNCParams& params = GNCParams());

}  // namespace gtsam_points
