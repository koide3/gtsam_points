#pragma once

#include <gtsam_ext/types/voxelized_frame.hpp>

#include <thrust/device_vector.h>

namespace gtsam_ext {

struct VoxelizedFrameGPU : public VoxelizedFrame {
public:
  VoxelizedFrameGPU(
    double voxel_resolution,
    const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs);

  std::unique_ptr<GaussianVoxelMapCPU> voxels_storage;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points_storage;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> covs_storage;

  thrust::device_vector<Eigen::Vector3f> points_gpu_storage;
  thrust::device_vector<Eigen::Matrix3f> covs_gpu_storage;
  std::unique_ptr<GaussianVoxelMapGPU> voxels_gpu_storage;
};

}  // namespace gtsam_ext
