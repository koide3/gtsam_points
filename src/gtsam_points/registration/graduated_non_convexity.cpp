// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#include <gtsam_points/registration/graduated_non_convexity.hpp>
#include <gtsam_points/registration/impl/graduated_non_convexity_impl.hpp>

#include <gtsam_points/types/point_cloud.hpp>

namespace gtsam_points {

#define DEFINE_GNC(PointCloud, Features)               \
  template RegistrationResult estimate_pose_gnc(       \
    const PointCloud& target,                          \
    const PointCloud& source,                          \
    const Features& target_features,                   \
    const Features& source_features,                   \
    const NearestNeighborSearch& target_tree,          \
    const NearestNeighborSearch& target_features_tree, \
    const NearestNeighborSearch& source_features_tree, \
    const GNCParams& params);

using PFHSignature = Eigen::Matrix<double, 125, 1>;
using FPFHSignature = Eigen::Matrix<double, 33, 1>;

DEFINE_GNC(PointCloud, std::vector<PFHSignature>)
DEFINE_GNC(PointCloud, std::vector<FPFHSignature>)
DEFINE_GNC(PointCloud, std::vector<Eigen::VectorXd>)

}  // namespace gtsam_points