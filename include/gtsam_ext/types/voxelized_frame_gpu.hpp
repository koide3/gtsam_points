#pragma once

#include <gtsam_ext/types/voxelized_frame.hpp>

#include <thrust/device_vector.h>

namespace gtsam_ext {

struct VoxelizedFrameGPU : public VoxelizedFrame {
public:
  using Ptr = std::shared_ptr<VoxelizedFrameGPU>;
  using ConstPtr = std::shared_ptr<const VoxelizedFrameGPU>;

  VoxelizedFrameGPU(
    double voxel_resolution,
    const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& points,
    const std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>>& covs);

  VoxelizedFrameGPU(
    double voxel_resolution,
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& points,
    const std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& covs);

  VoxelizedFrameGPU(
    double voxel_resolution,
    const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs);

  ~VoxelizedFrameGPU();

  // copy data from GPU to CPU
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> get_points_gpu() const;
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> get_covs_gpu() const;

  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> get_voxel_means_gpu() const;
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> get_voxel_covs_gpu() const;

private:
  void init(double voxel_resolution);

public:
  std::unique_ptr<GaussianVoxelMapCPU> voxels_storage;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points_storage;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> covs_storage;

  thrust::device_vector<Eigen::Vector3f> points_gpu_storage;
  thrust::device_vector<Eigen::Matrix3f> covs_gpu_storage;
  std::unique_ptr<GaussianVoxelMapGPU> voxels_gpu_storage;
};

}  // namespace gtsam_ext
