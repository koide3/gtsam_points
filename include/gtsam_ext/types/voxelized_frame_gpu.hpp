#pragma once

#include <gtsam_ext/types/voxelized_frame.hpp>

// forward declaration
namespace thrust {

template <typename T>
class device_allocator;

template <typename T, typename Alloc>
class device_vector;

}  // namespace thrust

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
  using PointsGPU = thrust::device_vector<Eigen::Vector3f, thrust::device_allocator<Eigen::Vector3f>>;
  using MatricesGPU = thrust::device_vector<Eigen::Matrix3f, thrust::device_allocator<Eigen::Matrix3f>>;

  std::unique_ptr<GaussianVoxelMapCPU> voxels_storage;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points_storage;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> covs_storage;

  std::unique_ptr<GaussianVoxelMapGPU> voxels_gpu_storage;
  std::unique_ptr<PointsGPU> points_gpu_storage;
  std::unique_ptr<MatricesGPU> covs_gpu_storage;
};

}  // namespace gtsam_ext
