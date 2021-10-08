#include <gtsam_ext/types/voxelized_frame_cpu.hpp>

#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>

namespace gtsam_ext {

VoxelizedFrameCPU::VoxelizedFrameCPU(
  double resolution,
  const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points_,
  const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs_) {
  //
  points_storage = points_;
  covs_storage = covs_;

  num_points = points_storage.size();
  points = &points_storage[0];
  covs = &covs_storage[0];

  voxels_storage.reset(new GaussianVoxelMapCPU(resolution));
  voxels_storage->create_voxelmap(*this);
  voxels = voxels_storage.get();
}

VoxelizedFrameCPU::~VoxelizedFrameCPU() {}

template <typename T>
void VoxelizedFrameCPU::add_times(const std::vector<T>& times) {
  assert(times.size() == size());
  times_storage.resize(times.size());
  std::transform(times.begin(), times.end(), times_storage.begin(), [](const auto& t) { return static_cast<double>(t); });
  this->times = times_storage.data();
}

template <typename T, int D>
void VoxelizedFrameCPU::add_normals(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& normals) {
  assert(normals.size() == size());
  normals_storage.resize(normals.size(), Eigen::Vector4d::Zero());
  for (int i = 0; i < normals.size(); i++) {
    normals_storage[i].head<D>() = normals[i].template head<D>().template cast<double>();
  }
  this->normals = normals_storage.data();
}

template void VoxelizedFrameCPU::add_times(const std::vector<float>& times);
template void VoxelizedFrameCPU::add_times(const std::vector<double>& times);
template void VoxelizedFrameCPU::add_normals(const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>& normals);
template void VoxelizedFrameCPU::add_normals(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>& normals);
template void VoxelizedFrameCPU::add_normals(const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>& normals);
template void VoxelizedFrameCPU::add_normals(const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>& normals);

}  // namespace gtsam_ext
