// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace gtsam_points {

/// @brief Registration result
struct RegistrationResult {
  double inlier_rate;                 ///< Inlier rate (The population differs for each method)
  Eigen::Isometry3d T_target_source;  ///< Estimated transformation
};

}  // namespace gtsam_points
