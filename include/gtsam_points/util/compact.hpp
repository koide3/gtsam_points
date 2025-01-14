// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)
#pragma once

#include <Eigen/Core>

namespace gtsam_points {

template <int D>
inline Eigen::Matrix<float, 6, 1> compact_cov(const Eigen::Matrix<double, D, D>& cov) {
  Eigen::Matrix<float, 6, 1> compact;
  compact << cov(0, 0), cov(1, 0), cov(1, 1), cov(2, 0), cov(2, 1), cov(2, 2);
  return compact;
}

inline Eigen::Matrix4d uncompact_cov(const Eigen::Matrix<float, 6, 1>& compact) {
  Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
  cov(0, 0) = compact(0);
  cov(1, 0) = cov(0, 1) = compact(1);
  cov(1, 1) = compact(2);
  cov(2, 0) = cov(0, 2) = compact(3);
  cov(2, 1) = cov(1, 2) = compact(4);
  cov(2, 2) = compact(5);
  return cov;
}

}  // namespace gtsam_points
