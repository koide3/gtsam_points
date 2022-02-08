// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam_ext/types/voxelized_frame.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_gpu.hpp>

namespace gtsam_ext {

struct VoxelizedFrameCPU : public VoxelizedFrame {
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

  template <typename T>
  void add_times(const T* times, int num_points);
  template <typename T>
  void add_times(const std::vector<T>& times) {
    add_times(times.data(), times.size());
  }

  template <typename T, int D>
  void add_points(const Eigen::Matrix<T, D, 1>* points, int num_points);
  template <typename T, int D, typename Alloc>
  void add_points(const std::vector<Eigen::Matrix<T, D, 1>, Alloc>& points) {
    add_points(points.data(), points.size());
  }

  template <typename T, int D>
  void add_normals(const Eigen::Matrix<T, D, 1>* normals, int num_points);
  template <typename T, int D, typename Alloc>
  void add_normals(const std::vector<Eigen::Matrix<T, D, 1>, Alloc>& normals) {
    add_normals(normals.data(), normals.size());
  }

  template <typename T, int D>
  void add_covs(const Eigen::Matrix<T, D, D>* covs, int num_points);
  template <typename T, int D, typename Alloc>
  void add_covs(const std::vector<Eigen::Matrix<T, D, D>, Alloc>& covs) {
    add_covs(covs.data(), covs.size());
  }

  template <typename T>
  void add_intensities(const T* intensities, int num_points);
  template <typename T>
  void add_intensities(const std::vector<T>& intensities) {
    add_intensities(intensities.data(), intensities.size());
  }

public:
  std::shared_ptr<GaussianVoxelMapCPU> voxels_storage;
  std::vector<double> times_storage;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points_storage;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> normals_storage;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> covs_storage;
  std::vector<double> intensities_storage;
};

}  // namespace gtsam_ext
