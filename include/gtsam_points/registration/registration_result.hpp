// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace gtsam_points {

struct RegistrationResult {
  double inlier_rate;
  Eigen::Isometry3d T_target_source;
};

}  // namespace gtsam_points
