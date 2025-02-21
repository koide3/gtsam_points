// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/registration/ransac.hpp>
#include <gtsam_points/registration/impl/ransac_impl.hpp>

#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/ann/kdtreex.hpp>
#include <gtsam_points/types/point_cloud.hpp>

#define DEFINE_RANSAC(PointCloud, Features)                               \
  template RegistrationResult estimate_pose_ransac_<PointCloud, Features>( \
    const PointCloud& target,                                             \
    const PointCloud& source,                                             \
    const Features& target_features,                                      \
    const Features& source_features,                                      \
    const NearestNeighborSearch& target_tree,                             \
    const NearestNeighborSearch& target_features_tree,                    \
    const RANSACParams& params);

namespace gtsam_points {

using PFHSignature = Eigen::Matrix<double, 125, 1>;
using FPFHSignature = Eigen::Matrix<double, 33, 1>;

using VextorXdPtr = const Eigen::VectorXd*;
using PFHSignaturePtr = const PFHSignature*;
using FPFHSignaturePtr = const FPFHSignature*;

DEFINE_RANSAC(PointCloud, VextorXdPtr);
DEFINE_RANSAC(PointCloud, FPFHSignaturePtr);
DEFINE_RANSAC(PointCloud, PFHSignaturePtr);

//
}  // namespace gtsam_points
