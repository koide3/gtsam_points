#include <gtsam_ext/types/voxelized_frame_cpu.hpp>

#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>

namespace gtsam_ext {

VoxelizedFrameCPU::VoxelizedFrameCPU(double voxel_resolution, const Eigen::Vector4d* points, const Eigen::Matrix4d* covs, int num_points) {
  points_storage.assign(points, points + num_points);
  covs_storage.assign(covs, covs + num_points);

  this->num_points = num_points;
  this->points = points_storage.data();
  this->covs = covs_storage.data();

  voxels_storage.reset(new GaussianVoxelMapCPU(voxel_resolution));
  voxels_storage->create_voxelmap(*this);
  voxels = voxels_storage;
}

VoxelizedFrameCPU::VoxelizedFrameCPU(
  double resolution,
  const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
  const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs)
: VoxelizedFrameCPU(resolution, points.data(), covs.data(), points.size()) {}

VoxelizedFrameCPU::VoxelizedFrameCPU(double voxel_resolution, const Frame::ConstPtr& frame) : VoxelizedFrameCPU(voxel_resolution, frame->points, frame->covs, frame->size()) {}

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

VoxelizedFrame::Ptr merge_voxelized_frames(
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& poses,
  const std::vector<Frame::ConstPtr>& frames,
  double downsample_resolution,
  double voxel_resolution) {
  //
  int num_all_points = 0;
  for (const auto& frame : frames) {
    num_all_points += frame->size();
  }

  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> all_points(num_all_points);
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> all_covs(num_all_points);

  int begin = 0;
  for (int i = 0; i < frames.size(); i++) {
    const auto& frame = frames[i];
    const auto& pose = poses[i];
    Eigen::Matrix4d R = Eigen::Matrix4d::Zero();
    R.block<3, 3>(0, 0) = pose.linear();

    for (int j = 0; j < frame->size(); j++) {
      all_points[j + begin] = pose * frame->points[j];
      all_covs[j + begin] = R * frame->covs[j] * R.transpose();
    }

    begin += frame->size();
  }

  Frame all_frames;
  all_frames.num_points = num_all_points;
  all_frames.points = all_points.data();
  all_frames.covs = all_covs.data();

  GaussianVoxelMapCPU downsampling(downsample_resolution);
  downsampling.create_voxelmap(all_frames);

  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> downsampled_points;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> downsampled_covs;
  downsampled_points.reserve(downsampling.voxels.size());
  downsampled_covs.reserve(downsampling.voxels.size());

  for (const auto& voxel : downsampling.voxels) {
    downsampled_points.push_back(voxel.second->mean);
    downsampled_covs.push_back(voxel.second->cov);
  }

  return VoxelizedFrame::Ptr(new VoxelizedFrameCPU(voxel_resolution, downsampled_points, downsampled_covs));
}

VoxelizedFrame::Ptr merge_voxelized_frames_auto(
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& poses,
  const std::vector<Frame::ConstPtr>& frames,
  double downsample_resolution,
  double voxel_resolution) {
//
#ifdef BUILD_GTSAM_EXT_GPU
  if (frames[0]->points_gpu) {
    return merge_voxelized_frames_gpu(poses, frames, downsample_resolution, voxel_resolution, true);
  }
#endif

  return merge_voxelized_frames(poses, frames, downsample_resolution, voxel_resolution);
}

}  // namespace gtsam_ext
