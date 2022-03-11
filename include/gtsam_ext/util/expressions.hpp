// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/expressions.h>
#include <gtsam/nonlinear/expressions.h>

namespace gtsam_ext {

namespace internal {
inline gtsam::Vector3 translation(const gtsam::Pose3& pose, gtsam::OptionalJacobian<3, 6> H) {
  return pose.translation(H);
}

template <int N>
inline Eigen::Matrix<double, N, 1> scale(const double s, const Eigen::Matrix<double, N, 1>& v, gtsam::OptionalJacobian<N, 1> H1, gtsam::OptionalJacobian<N, N> H2) {
  if (H1) {
    *H1 = v;
  }
  if (H2) {
    *H2 = s * Eigen::Matrix<double, N, N>::Identity();
  }
  return s * v;
}

inline gtsam::Pose3 se3(const gtsam::Rot3& rot, const gtsam::Vector3& trans, gtsam::OptionalJacobian<6, 3> H1, gtsam::OptionalJacobian<6, 3> H2) {
  if (H1) {
    H1->block<3, 3>(0, 0).setIdentity();
    H1->block<3, 3>(0, 3).setZero();
  }
  if (H2) {
    H2->block<3, 3>(0, 0).setZero();
    H2->block<3, 3>(0, 3).setIdentity();
  }

  return gtsam::Pose3(rot, trans);
}

template <int N>
struct vector_traits {
  static Eigen::Matrix<double, N, 1> Scale(const double s, const Eigen::Matrix<double, N, 1>& v, gtsam::OptionalJacobian<N, 1> H1, gtsam::OptionalJacobian<N, N> H2) {
    return scale(s, v, H1, H2);
  }
};

}  // namespace internal

inline gtsam::Vector3_ translation(const gtsam::Pose3_& x) {
  return gtsam::Vector3_(&internal::translation, x);
}

template <int N>
gtsam::Expression<Eigen::Matrix<double, N, 1>> scale(const gtsam::Double_& s, const gtsam::Expression<Eigen::Matrix<double, N, 1>>& v) {
  return gtsam::Expression<Eigen::Matrix<double, N, 1>>(&internal::vector_traits<N>::Scale, s, v);
}

inline gtsam::Pose3_ se3(const gtsam::Rot3_& rot, const gtsam::Vector3_& trans) {
  return gtsam::Pose3_(&internal::se3, rot, trans);
}

inline gtsam::Pose3_ expmap(const gtsam::Vector6_& x) {
  return gtsam::Pose3_(&gtsam::Pose3::Expmap, x);
}

inline gtsam::Vector6_ logmap(const gtsam::Pose3_& x) {
  return gtsam::Vector6_(&gtsam::Pose3::Logmap, x);
}

inline gtsam::Rot3_ expmap(const gtsam::Vector3_& x) {
  return gtsam::Rot3_(&gtsam::Rot3::Expmap, x);
}

inline gtsam::Pose3_ inverse(const gtsam::Pose3_& x) {
  auto f = [](const gtsam::Pose3& x, gtsam::OptionalJacobian<6, 6> H) { return x.inverse(H); };
  return gtsam::Pose3_(f, x);
}

}  // namespace gtsam_ext