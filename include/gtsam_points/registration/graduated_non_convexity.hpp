// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#pragma once

#include <random>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/ann/nearest_neighbor_search.hpp>
#include <gtsam_points/registration/registration_result.hpp>

namespace gtsam_points {

/// @brief Parameters for graduated non-convexity..
struct GNCParams {
public:
  // Feature matching parameters
  int max_init_samples = 5000;   ///< Maximum number of samples
  bool reciprocal_check = true;  ///< Reciprocal check
  bool tuple_check = false;      ///< Length similarity check
  double tuple_thresh = 0.9;     ///< Length similarity threshold
  int max_num_tuples = 1000;     ///< Number of tuples to be sampled

  // Estimation praameters
  double div_factor = 1.4;     ///< Division factor for graduated non-convexity
  double max_corr_dist = 0.25; ///< Maximum correspondence distance
  int innter_iterations = 3;   ///< Number of inner iterations
  int max_iterations = 64;     ///< Maximum number of iterations
  int dof = 6;                 ///< Degrees of freedom (must be 6 (SE3) or 4 (XYZ+RZ))

  // Misc
  bool verbose = false;        ///< Verbose mode
  std::uint64_t seed = 5489u;  ///< Random seed
  int num_threads = 4;         ///< Number of threads
};

/// @brief Fast global registration with graduated non-convexity
/// @ref   Zhou et al., "Fast Global Registration", ECCV2016
/// @param target                   Target point cloud
/// @param source                   Source point cloud
/// @param target_features          Target features
/// @param source_features          Source features
/// @param target_tree              Target nearest neighbor search
/// @param target_features_tree     Target features nearest neighbor search
/// @param source_features_tree     Source features nearest neighbor search
/// @param params                   GNC parameters
/// @return                         Registration result
template <typename PointCloud, typename Features>
RegistrationResult estimate_pose_gnc_(
  const PointCloud& target,
  const PointCloud& source,
  const Features& target_features,
  const Features& source_features,
  const NearestNeighborSearch& target_tree,
  const NearestNeighborSearch& target_features_tree,
  const NearestNeighborSearch& source_features_tree,
  const GNCParams& params = GNCParams());

//
RegistrationResult estimate_pose_gnc(
  const PointCloud& target,
  const PointCloud& source,
  const Eigen::Matrix<double, 33, 1>* target_features,
  const Eigen::Matrix<double, 33, 1>* source_features,
  const NearestNeighborSearch& target_tree,
  const NearestNeighborSearch& target_features_tree,
  const NearestNeighborSearch& source_features_tree,
  const GNCParams& params = GNCParams()) {
  using ConstFeaturePtr = const Eigen::Matrix<double, 33, 1>*;
  return estimate_pose_gnc_<PointCloud, ConstFeaturePtr>(
    target, source, target_features, source_features, target_tree, target_features_tree, source_features_tree, params);
}

RegistrationResult estimate_pose_gnc(
  const PointCloud& target,
  const PointCloud& source,
  const Eigen::Matrix<double, 125, 1>* target_features,
  const Eigen::Matrix<double, 125, 1>* source_features,
  const NearestNeighborSearch& target_tree,
  const NearestNeighborSearch& target_features_tree,
  const NearestNeighborSearch& source_features_tree,
  const GNCParams& params = GNCParams()) {
  using ConstFeaturePtr = const Eigen::Matrix<double, 125, 1>*;
  return estimate_pose_gnc_<PointCloud, ConstFeaturePtr>(
    target, source, target_features, source_features, target_tree, target_features_tree, source_features_tree, params);
}

RegistrationResult estimate_pose_gnc(
  const PointCloud& target,
  const PointCloud& source,
  const Eigen::VectorXd* target_features,
  const Eigen::VectorXd* source_features,
  const NearestNeighborSearch& target_tree,
  const NearestNeighborSearch& target_features_tree,
  const NearestNeighborSearch& source_features_tree,
  const GNCParams& params = GNCParams()) {
  using ConstFeaturePtr = const Eigen::VectorXd*;
  return estimate_pose_gnc_<PointCloud, ConstFeaturePtr>(
    target, source, target_features, source_features, target_tree, target_features_tree, source_features_tree, params);
}

}  // namespace gtsam_points
