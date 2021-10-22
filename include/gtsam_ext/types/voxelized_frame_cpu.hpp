#pragma once

#include <gtsam_ext/types/voxelized_frame.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_gpu.hpp>

namespace gtsam_ext {

struct VoxelizedFrameCPU : public VoxelizedFrame {
public:
  using Ptr = std::shared_ptr<VoxelizedFrameCPU>;
  using ConstPtr = std::shared_ptr<const VoxelizedFrameCPU>;

  VoxelizedFrameCPU(double voxel_resolution, const Eigen::Vector4d* points, const Eigen::Matrix4d* covs, int num_points);

  VoxelizedFrameCPU(
    double voxel_resolution,
    const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs);

  VoxelizedFrameCPU(double voxel_resolution, const Frame::ConstPtr& frame);

  ~VoxelizedFrameCPU();

  template <typename T>
  void add_times(const std::vector<T>& times);

  template <typename T, int D>
  void add_normals(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& normals);

  std::shared_ptr<GaussianVoxelMapCPU> voxels_storage;
  std::vector<double> times_storage;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points_storage;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> normals_storage;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> covs_storage;
};

}  // namespace gtsam_ext
