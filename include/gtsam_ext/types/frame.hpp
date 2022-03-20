// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtsam_ext/types/frame_traits.hpp>

namespace gtsam_ext {

struct VoxelizedFrame;

struct Frame {
public:
  using Ptr = std::shared_ptr<Frame>;
  using ConstPtr = std::shared_ptr<const Frame>;

  Frame()
  : num_points(0),
    times(nullptr),
    points(nullptr),
    normals(nullptr),
    covs(nullptr),
    intensities(nullptr),
    times_gpu(nullptr),
    points_gpu(nullptr),
    normals_gpu(nullptr),
    covs_gpu(nullptr),
    intensities_gpu(nullptr) {}
  virtual ~Frame() {}

  size_t size() const { return num_points; }

  // Check if the frame has attributes
  bool has_times() const;
  bool has_points() const;
  bool has_normals() const;
  bool has_covs() const;
  bool has_intensities() const;

  // Calculate the fraction of points fell in target's voxels
  // (evaluate if delta * this->points fall in target->voxels)
  double overlap(const std::shared_ptr<const VoxelizedFrame>& target, const Eigen::Isometry3d& delta) const;
  double overlap(
    const std::vector<std::shared_ptr<const VoxelizedFrame>>& targets,
    const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas) const;

  double overlap_gpu(const std::shared_ptr<const VoxelizedFrame>& target, const Eigen::Isometry3f* delta_gpu) const;
  double overlap_gpu(const std::shared_ptr<const VoxelizedFrame>& target, const Eigen::Isometry3d& delta) const;
  double overlap_gpu(
    const std::vector<std::shared_ptr<const VoxelizedFrame>>& targets,
    const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas) const;

  // Automatically select CPU or GPU method
  double overlap_auto(const std::shared_ptr<const VoxelizedFrame>& target, const Eigen::Isometry3d& delta) const;
  double overlap_auto(
    const std::vector<std::shared_ptr<const VoxelizedFrame>>& targets,
    const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas) const;

  void save(const std::string& path) const;
  void save_compact(const std::string& path) const;

public:
  size_t num_points;

  double* times;             // Time w.r.t. the first point (sorted)
  Eigen::Vector4d* points;   // Point coordinates (x, y, z, 1)
  Eigen::Vector4d* normals;  // Point normals (nx, ny, nz, 0)
  Eigen::Matrix4d* covs;     // Point covariances cov(3, 3) = 0
  double* intensities;       // Point intensities

  float* times_gpu;
  Eigen::Vector3f* points_gpu;
  Eigen::Vector3f* normals_gpu;
  Eigen::Matrix3f* covs_gpu;
  float* intensities_gpu;
};

namespace frame {

template <>
struct traits<Frame> {
  static int size(const Frame& frame) { return frame.size(); }

  static bool has_times(const Frame& frame) { return frame.has_times(); }
  static bool has_points(const Frame& frame) { return frame.has_points(); }
  static bool has_normals(const Frame& frame) { return frame.has_normals(); }
  static bool has_covs(const Frame& frame) { return frame.has_covs(); }
  static bool has_intensities(const Frame& frame) { return frame.has_intensities(); }

  static double time(const Frame& frame, size_t i) { return frame.times[i]; }
  static const Eigen::Vector4d& point(const Frame& frame, size_t i) { return frame.points[i]; }
  static const Eigen::Vector4d& normal(const Frame& frame, size_t i) { return frame.normals[i]; }
  static const Eigen::Matrix4d& cov(const Frame& frame, size_t i) { return frame.covs[i]; }
  static double intensity(const Frame& frame, size_t i) { return frame.intensities[i]; }

  static const Eigen::Vector4d* points_ptr(const Frame& frame) { return frame.points; }
};

}  // namespace frame
}  // namespace gtsam_ext