// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam_ext/types/frame_traits.hpp>

namespace gtsam_ext {

struct FrameCPU;

struct BasicFrame {
public:
  using Ptr = std::shared_ptr<BasicFrame>;
  using ConstPtr = std::shared_ptr<const BasicFrame>;

  BasicFrame() : num_points(0), times(nullptr), points(nullptr), normals(nullptr), covs(nullptr), intensities(nullptr) {}
  virtual ~BasicFrame() {}

  int size() const { return num_points; }

  bool has_times_() const { return times; }
  bool has_points_() const { return points; }
  bool has_normals_() const { return normals; }
  bool has_covs_() const { return covs; }
  bool has_intensities_() const { return intensities; }

  const double time(int i) const { return times[i]; }
  const Eigen::Vector4d& point(int i) const { return points[i]; }
  const Eigen::Vector4d& normal(int i) const { return normals[i]; }
  const Eigen::Matrix4d& cov(int i) const { return covs[i]; }
  const double intensity(int i) const { return intensities[i]; }

  const Eigen::Vector4d* points_ptr() const { return points; }

protected:
  size_t num_points;

  double* times;
  Eigen::Vector4d* points;
  Eigen::Vector4d* normals;
  Eigen::Matrix4d* covs;
  double* intensities;

  friend class FrameCPU;
};

namespace frame {

template <>
struct traits<BasicFrame> {
  static int size(const BasicFrame& frame) { return frame.size(); }

  static bool has_times(const BasicFrame& frame) { return frame.has_times_(); }
  static bool has_points(const BasicFrame& frame) { return frame.has_points_(); }
  static bool has_normals(const BasicFrame& frame) { return frame.has_normals_(); }
  static bool has_covs(const BasicFrame& frame) { return frame.has_covs_(); }
  static bool has_intensities(const BasicFrame& frame) { return frame.has_intensities_(); }

  static double time(const BasicFrame& frame, int i) { return frame.time(i); }
  static const Eigen::Vector4d& point(const BasicFrame& frame, int i) { return frame.point(i); }
  static const Eigen::Vector4d& normal(const BasicFrame& frame, int i) { return frame.normal(i); }
  static const Eigen::Matrix4d& cov(const BasicFrame& frame, int i) { return frame.cov(i); }
  static double intensity(const BasicFrame& frame, int i) { return frame.intensity(i); }

  static const Eigen::Vector4d* points_ptr(const BasicFrame& frame) { return frame.points_ptr(); }
};

}  // namespace frame

}  // namespace gtsam_ext