// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/expressions.h>
#include <gtsam/nonlinear/expressions.h>

namespace gtsam_points {

namespace internal {

template <int N>
struct vector_traits {
  static Eigen::Matrix<double, N, 1> Add(
    const Eigen::Matrix<double, N, 1>& v1,
    const Eigen::Matrix<double, N, 1>& v2,
    gtsam::OptionalJacobian<N, N> H1,
    gtsam::OptionalJacobian<N, N> H2) {
    if (H1) {
      H1->setIdentity();
    }
    if (H2) {
      H2->setIdentity();
    }
    return v1 + v2;
  }

  static Eigen::Matrix<double, N, 1>
  Scale(const double s, const Eigen::Matrix<double, N, 1>& v, gtsam::OptionalJacobian<N, 1> H1, gtsam::OptionalJacobian<N, N> H2) {
    if (H1) {
      *H1 = v;
    }
    if (H2) {
      *H2 = s * Eigen::Matrix<double, N, N>::Identity();
    }
    return s * v;
  }

  static Eigen::Matrix<double, N, 1> Product(
    const Eigen::Matrix<double, N, 1>& v1,
    const Eigen::Matrix<double, N, 1>& v2,
    gtsam::OptionalJacobian<N, N> H1,
    gtsam::OptionalJacobian<N, N> H2) {
    if (H1) {
      *H1 = v2.asDiagonal();
    }
    if (H2) {
      *H2 = v1.asDiagonal();
    }

    return v1.array() * v2.array();
  }
};

template <int N, int M>
struct vector2_traits {
  static Eigen::Matrix<double, N + M, 1> Concatenate(
    const Eigen::Matrix<double, N, 1>& x1,
    const Eigen::Matrix<double, M, 1>& x2,
    gtsam::OptionalJacobian<N + M, N> H1,
    gtsam::OptionalJacobian<N + M, M> H2) {
    if (H1) {
      H1->setZero();
      H1->template topLeftCorner<N, N>().setIdentity();
    }
    if (H2) {
      H2->setZero();
      H2->template bottomRightCorner<M, M>().setIdentity();
    }

    return (Eigen::Matrix<double, N + M, 1>() << x1, x2).finished();
  }
};

}  // namespace internal

template <int N>
gtsam::Expression<Eigen::Matrix<double, N, 1>> product(
  const gtsam::Expression<Eigen::Matrix<double, N, 1>>& v1,
  const gtsam::Expression<Eigen::Matrix<double, N, 1>>& v2) {
  return gtsam::Expression<Eigen::Matrix<double, N, 1>>(&internal::vector_traits<N>::Product, v1, v2);
}

template <int N>
gtsam::Expression<Eigen::Matrix<double, N, 1>> scale(const gtsam::Double_& s, const gtsam::Expression<Eigen::Matrix<double, N, 1>>& v) {
  return gtsam::Expression<Eigen::Matrix<double, N, 1>>(&internal::vector_traits<N>::Scale, s, v);
}

template <int N>
gtsam::Expression<Eigen::Matrix<double, N, 1>> add(
  const gtsam::Expression<Eigen::Matrix<double, N, 1>>& v1,
  const gtsam::Expression<Eigen::Matrix<double, N, 1>>& v2) {
  return gtsam::Expression<Eigen::Matrix<double, N, 1>>(&internal::vector_traits<N>::Add, v1, v2);
}

template <int N, int M>
gtsam::Expression<Eigen::Matrix<double, N + M, 1>> concatenate(
  const gtsam::Expression<Eigen::Matrix<double, N, 1>>& x1,
  const gtsam::Expression<Eigen::Matrix<double, M, 1>>& x2) {
  return gtsam::Expression<Eigen::Matrix<double, N + M, 1>>(&internal::vector2_traits<N, M>::Concatenate, x1, x2);
}

inline gtsam::Pose3_ create_se3(const gtsam::Rot3_& rot, const gtsam::Vector3_& trans) {
  return gtsam::Pose3_(&gtsam::Pose3::Create, rot, trans);
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

inline gtsam::Vector3_ logmap(const gtsam::Rot3_& x) {
  return gtsam::Vector3_(&gtsam::Rot3::Logmap, x);
}

inline gtsam::Pose3_ inverse(const gtsam::Pose3_& x) {
  auto f = [](const gtsam::Pose3& x, gtsam::OptionalJacobian<6, 6> H) { return x.inverse(H); };
  return gtsam::Pose3_(f, x);
}

}  // namespace gtsam_points