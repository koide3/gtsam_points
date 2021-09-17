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

}  // namespace gtsam_ext
