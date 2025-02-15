// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/registration/ransac.hpp>
#include <gtsam_points/registration/impl/ransac_impl.hpp>

#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/ann/kdtreex.hpp>
#include <gtsam_points/types/point_cloud.hpp>

#define DEFINE_RANSAC(PointCloud, Features)                               \
  template RegistrationResult estimate_pose_ransac<PointCloud, Features>( \
    const PointCloud& target,                                             \
    const PointCloud& source,                                             \
    const Features& target_features,                                      \
    const Features& source_features,                                      \
    const NearestNeighborSearch& target_tree,                             \
    const NearestNeighborSearch& target_features_tree,                    \
    double inlier_threshold,                                              \
    int max_iterations,                                                   \
    std::mt19937& mt);

namespace gtsam_points {

using PFHSignature = Eigen::Matrix<double, 125, 1>;
using FPFHSignature = Eigen::Matrix<double, 33, 1>;

DEFINE_RANSAC(PointCloud, std::vector<Eigen::VectorXd>);
DEFINE_RANSAC(PointCloud, std::vector<PFHSignature>);
DEFINE_RANSAC(PointCloud, std::vector<FPFHSignature>);

}  // namespace gtsam_points
