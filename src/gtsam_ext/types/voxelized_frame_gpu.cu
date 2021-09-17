#include <gtsam_ext/types/voxelized_frame_gpu.hpp>

#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_gpu.hpp>

namespace gtsam_ext {

VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points_,
  const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs_) {
  // CPU data
  points_storage = points_;
  covs_storage = covs_;

  num_points = points_storage.size();
  points = &points_storage[0];
  covs = &covs_storage[0];

  voxels_storage.reset(new GaussianVoxelMapCPU(voxel_resolution));
  voxels_storage->create_voxelmap(*this);
  voxels = voxels_storage.get();

  // GPU data
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> host_points(num_points);
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> host_covs(num_points);
  std::transform(points_.begin(), points_.end(), host_points.begin(), [](const Eigen::Vector4d& pt) { return pt.head<3>().cast<float>(); });
  std::transform(covs_.begin(), covs_.end(), host_covs.begin(), [](const Eigen::Matrix4d& cov) { return cov.block<3, 3>(0, 0).cast<float>(); });

  points_gpu_storage.resize(num_points);
  covs_gpu_storage.resize(num_points);
  cudaMemcpy(thrust::raw_pointer_cast(points_gpu_storage.data()), host_points.data(), sizeof(Eigen::Vector3f) * num_points, cudaMemcpyHostToDevice);
  cudaMemcpy(thrust::raw_pointer_cast(covs_gpu_storage.data()), host_covs.data(), sizeof(Eigen::Matrix3f) * num_points, cudaMemcpyHostToDevice);

  voxels_gpu_storage.reset(new GaussianVoxelMapGPU(voxel_resolution));
  voxels_gpu_storage->create_voxelmap(*this);
  voxels_gpu = voxels_gpu_storage.get();
}
}  // namespace gtsam_ext