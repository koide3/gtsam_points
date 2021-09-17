#pragma once

#include <gtsam_ext/types/voxelized_frame.hpp>

namespace gtsam_ext {

struct VoxelizedFrameCPU : public VoxelizedFrame {
public:
  VoxelizedFrameCPU(
    double voxel_resolution,
    const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs);

  std::unique_ptr<GaussianVoxelMapCPU> voxels_storage;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points_storage;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> covs_storage;
};

}  // namespace gtsam_ext
