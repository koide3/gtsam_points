// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <thrust/device_ptr.h>

namespace gtsam_points {

inline __host__ __device__ Eigen::Matrix3f skew_symmetric(const Eigen::Vector3f& x) {
  Eigen::Matrix3f skew = Eigen::Matrix3f::Zero();
  skew(0, 1) = -x[2];
  skew(0, 2) = x[1];
  skew(1, 0) = x[2];
  skew(1, 2) = -x[0];
  skew(2, 0) = -x[1];
  skew(2, 1) = x[0];

  return skew;
}

inline __host__ bool large_displacement(const Eigen::Isometry3f& x_p, const Eigen::Isometry3f& x, double angle_eps, double trans_eps) {
  Eigen::Isometry3f delta = x_p.inverse() * x;
  return std::abs(Eigen::AngleAxisf(delta.linear()).angle()) > angle_eps || delta.translation().norm() > trans_eps;
}

}  // namespace gtsam_points
