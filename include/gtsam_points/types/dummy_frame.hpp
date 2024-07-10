// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

namespace gtsam_points {

/// Dummy frame class for build test
struct DummyFrame {};

namespace frame {

template <>
struct traits<DummyFrame> {
  static int size(const DummyFrame& frame) { return 0; }

  static bool has_times(const DummyFrame& frame) { return false; }
  static bool has_points(const DummyFrame& frame) { return false; }
  static bool has_normals(const DummyFrame& frame) { return false; }
  static bool has_covs(const DummyFrame& frame) { return false; }
  static bool has_intensities(const DummyFrame& frame) { return false; }

  static double time(const DummyFrame& frame, int i) { return 0.0; }
  static const Eigen::Vector4d point(const DummyFrame& frame, int i) { return Eigen::Vector4d(0, 0, 0, 1); }
  static const Eigen::Vector4d normal(const DummyFrame& frame, int i) { return Eigen::Vector4d(0, 0, 0, 0); }
  static const Eigen::Matrix4d cov(const DummyFrame& frame, int i) { return Eigen::Matrix4d::Zero(); }
  static double intensity(const DummyFrame& frame, int i) { return 0.0; }

  static const Eigen::Vector4d* points_ptr(const DummyFrame& frame) { return nullptr; }
};
}  // namespace frame
}  // namespace gtsam_points