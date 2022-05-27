// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <iostream>
#include <gtsam_ext/types/frame.hpp>

#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_gpu.hpp>

namespace gtsam_ext {

// class GaussianVoxelMapCPU;
// class GaussianVoxelMapGPU;

/**
 * @brief Voxelized point cloud frame
 */
struct VoxelizedFrame : public Frame {
public:
  using Ptr = std::shared_ptr<VoxelizedFrame>;
  using ConstPtr = std::shared_ptr<const VoxelizedFrame>;

  VoxelizedFrame() : voxels(nullptr), voxels_gpu(nullptr) {}
  virtual ~VoxelizedFrame() {}

  /// Voxel resolution
  double voxel_resolution() const {
    if (voxels) {
      return voxels->voxel_resolution();
    } else if (voxels_gpu) {
      return voxels_gpu->voxel_resolution();
    }

    std::cerr << "warning: CPU/GPU voxels have not been created and failed to get voxel resolution!!" << std::endl;
    return -1.0;
  }

public:
  std::shared_ptr<GaussianVoxelMapCPU> voxels;      ///< Voxelmap on CPU
  std::shared_ptr<GaussianVoxelMapGPU> voxels_gpu;  ///< Voxelmap on GPU
};

/**
 * @brief Merge a set of voxelized frames into one frame
 * @note  This function only merges points and covs and discard other point attributes.
 * @param poses                  Poses of input frames
 * @param frames                 Input frames
 * @param downsample_resolution  Downsampling resolution
 * @param voxel_resolution       Voxel resolution
 * @return                       Merged frame
 */
VoxelizedFrame::Ptr merge_voxelized_frames(
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& poses,
  const std::vector<Frame::ConstPtr>& frames,
  double downsample_resolution,
  double voxel_resolution);

VoxelizedFrame::Ptr merge_voxelized_frames_gpu(
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& poses,
  const std::vector<Frame::ConstPtr>& frames,
  double downsample_resolution,
  double voxel_resolution,
  bool allocate_cpu);

VoxelizedFrame::Ptr merge_voxelized_frames_auto(
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& poses,
  const std::vector<Frame::ConstPtr>& frames,
  double downsample_resolution,
  double voxel_resolution);

/**
 * @brief Calculate the fraction of points fell in target's voxels
 * @param target   Target voxelized frame
 * @param source   Source frame
 * @param delta    T_target_source
 * @return         Overlap rate
 */
double overlap(const GaussianVoxelMap::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta);

double overlap(const VoxelizedFrame::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta);

/**
 * @brief Calculate the fraction of points fell in targets' voxels
 * @param target   Set of target voxelized frames
 * @param source   Source frame
 * @param delta    Set of T_target_source
 * @return         Overlap rate
 */
double overlap(
  const std::vector<GaussianVoxelMap::ConstPtr>& targets,
  const Frame::ConstPtr& source,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas);

double overlap(
  const std::vector<VoxelizedFrame::ConstPtr>& targets,
  const Frame::ConstPtr& source,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas);

/**
 * @brief Calculate the fraction of points fell in target voxels on GPU
 * @note  Source points and target voxelmap must be pre-allocated on GPU.
 * @param target     Target voxelmap
 * @param source     Source frame
 * @param delta_gpu  T_target_source (on GPU memory)
 * @return         Overlap rate
 */
double overlap_gpu(const GaussianVoxelMap::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3f* delta_gpu);

/// @brief Calculate the fraction of points fell in target voxels on GPU
double overlap_gpu(const VoxelizedFrame::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3f* delta_gpu);

/**
 * @brief Calculate the fraction of points fell in target voxels on GPU
 * @note  Source points and target voxelmap must be pre-allocated on GPU.
 * @param target  Target voxelized frame
 * @param source  Source frame
 * @param delta   T_target_source
 * @return         Overlap rate
 */
double overlap_gpu(const GaussianVoxelMap::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta);

/// @brief Calculate the fraction of points fell in target voxels on GPU
double overlap_gpu(const VoxelizedFrame::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta);

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
  const Frame::ConstPtr& source,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas);

/// @brief Calculate the fraction of points fell in targets' voxels on GPU
double overlap_gpu(
  const std::vector<VoxelizedFrame::ConstPtr>& targets,
  const Frame::ConstPtr& source,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas);

// Automatically select CPU or GPU method
double overlap_auto(const GaussianVoxelMap::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta);

double overlap_auto(const VoxelizedFrame::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta);

double overlap_auto(
  const std::vector<GaussianVoxelMap::ConstPtr>& targets,
  const Frame::ConstPtr& source,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas);

double overlap_auto(
  const std::vector<VoxelizedFrame::ConstPtr>& targets,
  const Frame::ConstPtr& source,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas);

}  // namespace gtsam_ext
