// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam_ext/types/frame_cpu.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>

struct CUstream_st;

namespace gtsam_ext {

struct VoxelizedFrameCPU : public FrameCPU {
public:
  using Ptr = std::shared_ptr<VoxelizedFrameCPU>;
  using ConstPtr = std::shared_ptr<const VoxelizedFrameCPU>;

  template <typename T, int D>
  VoxelizedFrameCPU(double voxel_resolution, const Eigen::Matrix<T, D, 1>* points, const Eigen::Matrix<T, D, D>* covs, int num_points);

  template <typename T, int D, template <class> class Alloc>
  VoxelizedFrameCPU(
    double voxel_resolution,
    const std::vector<Eigen::Matrix<T, D, 1>, Alloc<Eigen::Matrix<T, D, 1>>>& points,
    const std::vector<Eigen::Matrix<T, D, D>, Alloc<Eigen::Matrix<T, D, D>>>& covs)
  : VoxelizedFrameCPU(voxel_resolution, points.data(), covs.data(), points.size()) {}

  VoxelizedFrameCPU(double voxel_resolution, const Frame& frame);

  VoxelizedFrameCPU();
  ~VoxelizedFrameCPU();

  void create_voxelmap(double voxel_resolution);
};

/**
 * @brief Calculate the fraction of points fell in target's voxels
 * @param target   Target voxelized frame
 * @param source   Source frame
 * @param delta    T_target_source
 * @return         Overlap rate
 */
double overlap(const GaussianVoxelMap::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta);

double overlap(const Frame::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta);

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
  const std::vector<Frame::ConstPtr>& targets,
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
double
overlap_gpu(const GaussianVoxelMap::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3f* delta_gpu, CUstream_st* stream = 0);

/// @brief Calculate the fraction of points fell in target voxels on GPU
double overlap_gpu(const Frame::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3f* delta_gpu, CUstream_st* stream = 0);

/**
 * @brief Calculate the fraction of points fell in target voxels on GPU
 * @note  Source points and target voxelmap must be pre-allocated on GPU.
 * @param target  Target voxelized frame
 * @param source  Source frame
 * @param delta   T_target_source
 * @return         Overlap rate
 */
double overlap_gpu(const GaussianVoxelMap::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta, CUstream_st* stream = 0);

/// @brief Calculate the fraction of points fell in target voxels on GPU
double overlap_gpu(const Frame::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta, CUstream_st* stream = 0);

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
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas,
  CUstream_st* stream = 0);

/// @brief Calculate the fraction of points fell in targets' voxels on GPU
double overlap_gpu(
  const std::vector<Frame::ConstPtr>& targets,
  const Frame::ConstPtr& source,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas,
  CUstream_st* stream = 0);

// Automatically select CPU or GPU method
double overlap_auto(const GaussianVoxelMap::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta);

double overlap_auto(const Frame::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta);

double overlap_auto(
  const std::vector<GaussianVoxelMap::ConstPtr>& targets,
  const Frame::ConstPtr& source,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas);

double overlap_auto(
  const std::vector<Frame::ConstPtr>& targets,
  const Frame::ConstPtr& source,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas);

}  // namespace gtsam_ext
