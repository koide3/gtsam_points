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

struct VoxelizedFrame : public Frame {
public:
  using Ptr = std::shared_ptr<VoxelizedFrame>;
  using ConstPtr = std::shared_ptr<const VoxelizedFrame>;

  VoxelizedFrame() : voxels(nullptr), voxels_gpu(nullptr) {}
  virtual ~VoxelizedFrame() {}

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
  std::shared_ptr<GaussianVoxelMapCPU> voxels;
  std::shared_ptr<GaussianVoxelMapGPU> voxels_gpu;
};

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

// Calculate the fraction of points fell in target's voxels
// (evaluate if delta * this->points fall in target->voxels)
double overlap(const VoxelizedFrame::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta);
double overlap(
  const std::vector<VoxelizedFrame::ConstPtr>& targets,
  const Frame::ConstPtr& source,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas);

double overlap_gpu(const VoxelizedFrame::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3f* delta_gpu);
double overlap_gpu(const VoxelizedFrame::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta);
double overlap_gpu(
  const std::vector<VoxelizedFrame::ConstPtr>& targets,
  const Frame::ConstPtr& source,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas);

// Automatically select CPU or GPU method
double overlap_auto(const VoxelizedFrame::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta);
double overlap_auto(
  const std::vector<VoxelizedFrame::ConstPtr>& targets,
  const Frame::ConstPtr& source,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas);

}  // namespace gtsam_ext
