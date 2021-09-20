#include <gtsam_ext/types/voxelized_frame_gpu.hpp>

#include <thrust/device_vector.h>
#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_gpu.hpp>

namespace gtsam_ext {

// note: (points|covs)_storage must be allocated before calling this
void VoxelizedFrameGPU::init(double voxel_resolution) {
  // CPU data
  num_points = points_storage.size();
  points = &points_storage[0];
  covs = &covs_storage[0];

  voxels_storage.reset(new GaussianVoxelMapCPU(voxel_resolution));
  voxels_storage->create_voxelmap(*this);
  voxels = voxels_storage.get();

  // GPU data
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> host_points(num_points);
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> host_covs(num_points);
  std::transform(points_storage.begin(), points_storage.end(), host_points.begin(), [](const Eigen::Vector4d& pt) { return pt.head<3>().cast<float>(); });
  std::transform(covs_storage.begin(), covs_storage.end(), host_covs.begin(), [](const Eigen::Matrix4d& cov) { return cov.block<3, 3>(0, 0).cast<float>(); });

  points_gpu_storage.reset(new PointsGPU);
  covs_gpu_storage.reset(new MatricesGPU);

  points_gpu_storage->resize(num_points);
  covs_gpu_storage->resize(num_points);
  cudaMemcpy(thrust::raw_pointer_cast(points_gpu_storage->data()), host_points.data(), sizeof(Eigen::Vector3f) * num_points, cudaMemcpyHostToDevice);
  cudaMemcpy(thrust::raw_pointer_cast(covs_gpu_storage->data()), host_covs.data(), sizeof(Eigen::Matrix3f) * num_points, cudaMemcpyHostToDevice);

  points_gpu = thrust::raw_pointer_cast(points_gpu_storage->data());
  covs_gpu = thrust::raw_pointer_cast(covs_gpu_storage->data());

  voxels_gpu_storage.reset(new GaussianVoxelMapGPU(voxel_resolution));
  voxels_gpu_storage->create_voxelmap(*this);
  voxels_gpu = voxels_gpu_storage.get();
}

VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& points_,
  const std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>>& covs_) {
  // CPU data
  points_storage.resize(points_.size());
  covs_storage.resize(covs_.size());
  for (int i = 0; i < points_.size(); i++) {
    points_storage[i] << points_[i].cast<double>(), 1.0f;
    covs_storage[i].setZero();
    covs_storage[i].block<3, 3>(0, 0) = covs_[i].cast<double>();
  }

  init(voxel_resolution);
}

VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& points_,
  const std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& covs_) {
  // CPU data
  points_storage.resize(points_.size());
  covs_storage.resize(covs_.size());
  for (int i = 0; i < points_.size(); i++) {
    points_storage[i] << points_[i], 1.0;
    covs_storage[i].setZero();
    covs_storage[i].block<3, 3>(0, 0) = covs_[i];
  }

  init(voxel_resolution);
}

VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points_,
  const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs_) {
  // CPU data
  points_storage = points_;
  covs_storage = covs_;

  init(voxel_resolution);
}

VoxelizedFrameGPU::~VoxelizedFrameGPU() {}

std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> VoxelizedFrameGPU::get_points_gpu() const {
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> buffer(points_gpu_storage->size());
  cudaMemcpy(buffer.data(), thrust::raw_pointer_cast(points_gpu_storage->data()), sizeof(Eigen::Vector3f) * points_gpu_storage->size(), cudaMemcpyDeviceToHost);
  return buffer;
}

std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> VoxelizedFrameGPU::get_covs_gpu() const {
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> buffer(covs_gpu_storage->size());
  cudaMemcpy(buffer.data(), thrust::raw_pointer_cast(covs_gpu_storage->data()), sizeof(Eigen::Matrix3f) * covs_gpu_storage->size(), cudaMemcpyDeviceToHost);
  return buffer;
}

std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> VoxelizedFrameGPU::get_voxel_means_gpu() const {
  const auto& means_storage = *(voxels_gpu->voxel_means);
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> buffer(means_storage.size());
  cudaMemcpy(buffer.data(), thrust::raw_pointer_cast(means_storage.data()), sizeof(Eigen::Vector3f) * means_storage.size(), cudaMemcpyDeviceToHost);
  return buffer;
}

std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> VoxelizedFrameGPU::get_voxel_covs_gpu() const {
  const auto& covs_storage = *(voxels_gpu->voxel_covs);
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> buffer(covs_storage.size());
  cudaMemcpy(buffer.data(), thrust::raw_pointer_cast(covs_storage.data()), sizeof(Eigen::Matrix3f) * covs_storage.size(), cudaMemcpyDeviceToHost);
  return buffer;
}

}  // namespace gtsam_ext