// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam_ext/types/frame_gpu.hpp>
#include <gtsam_ext/types/voxelized_frame_cpu.hpp>

namespace gtsam_ext {

struct VoxelizedFrameGPU : public FrameGPU {
public:
  using Ptr = std::shared_ptr<VoxelizedFrameGPU>;
  using ConstPtr = std::shared_ptr<const VoxelizedFrameGPU>;

  template <typename T, int D>
  VoxelizedFrameGPU(double voxel_resolution, const Eigen::Matrix<T, D, 1>* points, const Eigen::Matrix<T, D, D>* covs, int num_points);

  template <typename T, int D, template <typename> typename Alloc>
  VoxelizedFrameGPU(
    double voxel_resolution,
    const std::vector<Eigen::Matrix<T, D, 1>, Alloc<Eigen::Matrix<T, D, 1>>>& points,
    const std::vector<Eigen::Matrix<T, D, D>, Alloc<Eigen::Matrix<T, D, D>>>& covs)
  : VoxelizedFrameGPU(voxel_resolution, points.data(), covs.data(), points.size()) {}

  VoxelizedFrameGPU(double voxel_resolution, const Frame& frame);
  VoxelizedFrameGPU();
  ~VoxelizedFrameGPU();

  void create_voxelmap(double voxel_resolution);
  void create_voxelmap_gpu(double voxel_resolution);

  // copy data from GPU to CPU
  std::vector<std::pair<Eigen::Vector3i, int>> get_voxel_buckets_gpu() const;
  std::vector<int> get_voxel_num_points_gpu() const;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> get_voxel_means_gpu() const;
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> get_voxel_covs_gpu() const;
};

}  // namespace gtsam_ext
