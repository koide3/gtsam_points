// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace gtsam_points {

/// @brief  Find the 6-DoF transformation (SE3) that aligns three point pairs.
/// @return T_target_source
Eigen::Isometry3d align_points(
  const Eigen::Vector4d& target1,
  const Eigen::Vector4d& target2,
  const Eigen::Vector4d& target3,
  const Eigen::Vector4d& source1,
  const Eigen::Vector4d& source2,
  const Eigen::Vector4d& source3);

/// @brief  Find the 4-DoF transformation (XYZ + RZ) that aligns three point pairs.
/// @return T_target_source
Eigen::Isometry3d
align_points_4dof(const Eigen::Vector4d& target1, const Eigen::Vector4d& target2, const Eigen::Vector4d& source1, const Eigen::Vector4d& source2);

namespace impl {

double sum_diffs(const Eigen::Isometry3d& T_target_source, const Eigen::Vector4d& target, const Eigen::Vector4d& source) {
  return (target - T_target_source * source).squaredNorm();
}

template <typename... Rest>
double sum_diffs(const Eigen::Isometry3d& T_target_source, const Eigen::Vector4d& target, const Eigen::Vector4d& source, const Rest&... rest) {
  return (target - T_target_source * source).squaredNorm() + sum_diffs(T_target_source, rest...);
}

template <typename... Args>
double sum_sq_errors(const Eigen::Isometry3d& T_target_source, const Args&... args) {
  static_assert(sizeof...(Args) % 2 == 0, "number of arguments must be even");
  return sum_diffs(T_target_source, args...);
}

}  // namespace impl

}  // namespace gtsam_points
