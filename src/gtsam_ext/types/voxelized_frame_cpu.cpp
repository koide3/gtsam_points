#include <gtsam_ext/types/voxelized_frame_cpu.hpp>

#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>

namespace gtsam_ext {

template <typename T, int D>
VoxelizedFrameCPU::VoxelizedFrameCPU(double voxel_resolution, const Eigen::Matrix<T, D, 1>* points, const Eigen::Matrix<T, D, D>* covs, int num_points) {
  add_points(points, num_points);
  add_covs(covs, num_points);
  create_voxelmap(voxel_resolution);
}

template <typename T, int D>
VoxelizedFrameCPU::VoxelizedFrameCPU(
  double voxel_resolution,
  const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& points,
  const std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>>& covs)
: VoxelizedFrameCPU(voxel_resolution, points.data(), covs.data(), points.size()) {}

VoxelizedFrameCPU::VoxelizedFrameCPU(double voxel_resolution, const Frame& frame) {
  if (!frame.points) {
    std::cerr << "error: input frame doesn't have points!!" << std::endl;
    abort();
  }

  if (!frame.covs) {
    std::cerr << "error: input frame doesn't have covs!!" << std::endl;
    abort();
  }

  add_points(frame.points, frame.size());
  add_covs(frame.covs, frame.size());
  create_voxelmap(voxel_resolution);

  if (frame.times) {
    add_times(frame.times, frame.size());
  }

  if (frame.normals) {
    add_normals(frame.normals, frame.size());
  }
}

VoxelizedFrameCPU::VoxelizedFrameCPU() {}

VoxelizedFrameCPU::~VoxelizedFrameCPU() {}

void VoxelizedFrameCPU::create_voxelmap(double voxel_resolution) {
  if (!points) {
    std::cerr << "error: frame doesn't have points!!" << std::endl;
    abort();
  }

  if (!covs) {
    std::cerr << "error: frame doesn't have covs!!" << std::endl;
    abort();
  }

  voxels_storage.reset(new GaussianVoxelMapCPU(voxel_resolution));
  voxels_storage->create_voxelmap(*this);
  voxels = voxels_storage;
}

template <typename T>
void VoxelizedFrameCPU::add_times(const T* times, int num_points) {
  assert(num_points == size());
  times_storage.resize(num_points);
  std::copy(times, times + num_points, times_storage.begin());
  this->times = this->times_storage.data();
}

template <typename T>
void VoxelizedFrameCPU::add_times(const std::vector<T>& times) {
  add_times(times.data(), times.size());
}

template <typename T, int D>
void VoxelizedFrameCPU::add_points(const Eigen::Matrix<T, D, 1>* points, int num_points) {
  points_storage.resize(num_points, Eigen::Vector4d(0.0, 0.0, 0.0, 1.0));
  for (int i = 0; i < num_points; i++) {
    points_storage[i].head<D>() = points[i].template head<D>().template cast<double>();
  }
  this->points = points_storage.data();
  this->num_points = num_points;
}

template <typename T, int D>
void VoxelizedFrameCPU::add_points(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& points) {
  add_points(points.data(), points.size());
}

template <typename T, int D>
void VoxelizedFrameCPU::add_normals(const Eigen::Matrix<T, D, 1>* normals, int num_points) {
  assert(num_points == size());
  normals_storage.resize(num_points, Eigen::Vector4d::Zero());
  for (int i = 0; i < num_points; i++) {
    normals_storage[i].head<D>() = normals[i].template head<D>().template cast<double>();
  }
  this->normals = normals_storage.data();
}

template <typename T, int D>
void VoxelizedFrameCPU::add_normals(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& normals) {
  add_normals(normals.data(), normals.size());
}

template <typename T, int D>
void VoxelizedFrameCPU::add_covs(const Eigen::Matrix<T, D, D>* covs, int num_points) {
  assert(num_points == size());
  covs_storage.resize(num_points, Eigen::Matrix4d::Zero());
  for (int i = 0; i < num_points; i++) {
    covs_storage[i].block<D, D>(0, 0) = covs[i].template block<D, D>(0, 0).template cast<double>();
  }
  this->covs = covs_storage.data();
}

template <typename T, int D>
void VoxelizedFrameCPU::add_covs(const std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>>& covs) {
  add_covs(covs.data(), covs.size());
}

template VoxelizedFrameCPU::VoxelizedFrameCPU(
  double voxel_resolution,
  const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>&,
  const std::vector<Eigen::Matrix<float, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 3>>>&);
template VoxelizedFrameCPU::VoxelizedFrameCPU(
  double voxel_resolution,
  const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>&,
  const std::vector<Eigen::Matrix<float, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 4>>>&);
template VoxelizedFrameCPU::VoxelizedFrameCPU(
  double voxel_resolution,
  const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>&,
  const std::vector<Eigen::Matrix<double, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 3>>>&);
template VoxelizedFrameCPU::VoxelizedFrameCPU(
  double voxel_resolution,
  const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>&,
  const std::vector<Eigen::Matrix<double, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>>&);

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
