// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam_points/types/point_cloud.hpp>

// forward declaration
struct CUstream_st;

namespace gtsam_points {

/**
 * @brief Gaussian distribution voxelmap
 */
class GaussianVoxelMap {
public:
  using Ptr = std::shared_ptr<GaussianVoxelMap>;
  using ConstPtr = std::shared_ptr<const GaussianVoxelMap>;

  GaussianVoxelMap() {}
  virtual ~GaussianVoxelMap() {}

  /// Voxel resolution
  virtual double voxel_resolution() const = 0;

  /// Insert a point cloud frame into the voxelmap
  virtual void insert(const PointCloud& frame) = 0;
};

/**
 * @brief Calculate the fraction of points fell in target's voxels
 * @param target            Target voxelized frame
 * @param source            Source frame
 * @param T_target_source   T_target_source
 * @return                  Overlap rate
 */
double overlap(const GaussianVoxelMap::ConstPtr& target, const PointCloud::ConstPtr& source, const Eigen::Isometry3d& T_target_source);

/**
 * @brief Calculate the fraction of points fell in targets' voxels
 * @param target   Set of target voxelized frames
 * @param source   Source frame
 * @param delta    Set of T_target_source
 * @return         Overlap rate
 */
double overlap(
  const std::vector<GaussianVoxelMap::ConstPtr>& targets,
  const PointCloud::ConstPtr& source,
  const std::vector<Eigen::Isometry3d>& Ts_target_source);

/// @brief Calculate the fraction of points fell in targets' voxels
template <typename VoxelMapPtr>
std::enable_if_t<!std::is_same_v<VoxelMapPtr, GaussianVoxelMap::ConstPtr>, double>
overlap(const std::vector<VoxelMapPtr>& targets, const PointCloud::ConstPtr& source, const std::vector<Eigen::Isometry3d>& Ts_target_source) {
  const std::vector<GaussianVoxelMap::ConstPtr> targets_(targets.begin(), targets.end());
  return overlap(targets_, source, Ts_target_source);
}

/**
 * @brief Calculate the fraction of points fell in target voxels on GPU
 * @note  Source points and target voxelmap must be pre-allocated on GPU.
 * @param target     Target voxelmap
 * @param source     Source frame
 * @param delta_gpu  T_target_source (on GPU memory)
 * @return         Overlap rate
 */
double overlap_gpu(
  const GaussianVoxelMap::ConstPtr& target,
  const PointCloud::ConstPtr& source,
  const Eigen::Isometry3f* delta_gpu,
  CUstream_st* stream = 0);

/**
 * @brief Calculate the fraction of points fell in target voxels on GPU
 * @note  Source points and target voxelmap must be pre-allocated on GPU.
 * @param target  Target voxelmap
 * @param source  Source frame
 * @param delta   T_target_source
 * @return         Overlap rate
 */
double
overlap_gpu(const GaussianVoxelMap::ConstPtr& target, const PointCloud::ConstPtr& source, const Eigen::Isometry3d& delta, CUstream_st* stream = 0);

/// @brief Calculate the fraction of points fell in target voxels on GPU
double overlap_gpu(const PointCloud::ConstPtr& target, const PointCloud::ConstPtr& source, const Eigen::Isometry3d& delta, CUstream_st* stream = 0);

/**
 * @brief Calculate the fraction of points fell in targets' voxels on GPU
 * @note  Source points and targets' voxelmap must be pre-allocated on GPU.
 * @param targets Set of target voxelized frames
 * @param source  Source frame
 * @param deltas  Set of T_target_source
 * @return         Overlap rate
 */
double overlap_gpu(
  const std::vector<GaussianVoxelMap::ConstPtr>& targets,
  const PointCloud::ConstPtr& source,
  const std::vector<Eigen::Isometry3d>& deltas,
  CUstream_st* stream = 0);

/// @brief Calculate the fraction of points fell in targets' voxels on GPU
template <typename VoxelMapPtr>
std::enable_if_t<!std::is_same_v<VoxelMapPtr, GaussianVoxelMap::ConstPtr>, double> overlap_gpu(
  const std::vector<VoxelMapPtr>& targets,
  const PointCloud::ConstPtr& source,
  const std::vector<Eigen::Isometry3d>& deltas,
  CUstream_st* stream = 0) {
  const std::vector<GaussianVoxelMap::ConstPtr> targets_(targets.begin(), targets.end());
  return overlap_gpu(targets_, source, deltas, stream);
}

// Automatically select CPU or GPU method
double overlap_auto(const GaussianVoxelMap::ConstPtr& target, const PointCloud::ConstPtr& source, const Eigen::Isometry3d& delta);

double overlap_auto(
  const std::vector<GaussianVoxelMap::ConstPtr>& targets,
  const PointCloud::ConstPtr& source,
  const std::vector<Eigen::Isometry3d>& deltas);

template <typename VoxelMapPtr>
std::enable_if_t<!std::is_same_v<VoxelMapPtr, GaussianVoxelMap::ConstPtr>, double>
overlap_auto(const std::vector<VoxelMapPtr>& targets, const PointCloud::ConstPtr& source, const std::vector<Eigen::Isometry3d>& deltas) {
  const std::vector<GaussianVoxelMap::ConstPtr> targets_(targets.begin(), targets.end());
  return overlap_auto(targets_, source, deltas);
}

}  // namespace gtsam_points