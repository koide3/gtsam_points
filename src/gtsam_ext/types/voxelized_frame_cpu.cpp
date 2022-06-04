// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_ext/types/voxelized_frame_cpu.hpp>

#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_gpu.hpp>

namespace gtsam_ext {

// constructors & destructor
template <typename T, int D>
VoxelizedFrameCPU::VoxelizedFrameCPU(
  double voxel_resolution,
  const Eigen::Matrix<T, D, 1>* points,
  const Eigen::Matrix<T, D, D>* covs,
  int num_points) {
  add_points(points, num_points);
  add_covs(covs, num_points);
  create_voxelmap(voxel_resolution);
}

template VoxelizedFrameCPU::VoxelizedFrameCPU(
  double voxel_resolution,
  const Eigen::Matrix<float, 3, 1>* points,
  const Eigen::Matrix<float, 3, 3>* covs,
  int num_points);
template VoxelizedFrameCPU::VoxelizedFrameCPU(
  double voxel_resolution,
  const Eigen::Matrix<float, 4, 1>* points,
  const Eigen::Matrix<float, 4, 4>* covs,
  int num_points);
template VoxelizedFrameCPU::VoxelizedFrameCPU(
  double voxel_resolution,
  const Eigen::Matrix<double, 3, 1>* points,
  const Eigen::Matrix<double, 3, 3>* covs,
  int num_points);
template VoxelizedFrameCPU::VoxelizedFrameCPU(
  double voxel_resolution,
  const Eigen::Matrix<double, 4, 1>* points,
  const Eigen::Matrix<double, 4, 4>* covs,
  int num_points);

VoxelizedFrameCPU::VoxelizedFrameCPU(double voxel_resolution, const Frame& frame) : FrameCPU(frame) {
  if (!frame.points) {
    std::cerr << "error: input frame doesn't have points!!" << std::endl;
    abort();
  }

  if (!frame.covs) {
    std::cerr << "error: input frame doesn't have covs!!" << std::endl;
    abort();
  }

  create_voxelmap(voxel_resolution);
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

  voxels.reset(new GaussianVoxelMapCPU(voxel_resolution));
  voxels->insert(*this);
}

// merge_frames
Frame::Ptr merge_frames(
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
    for (int j = 0; j < frame->size(); j++) {
      all_points[j + begin] = pose * frame->points[j];
      all_covs[j + begin] = pose.matrix() * frame->covs[j] * pose.matrix().transpose();
    }

    begin += frame->size();
  }

  Frame all_frames;
  all_frames.num_points = num_all_points;
  all_frames.points = all_points.data();
  all_frames.covs = all_covs.data();

  GaussianVoxelMapCPU downsampling(downsample_resolution);
  downsampling.insert(all_frames);

  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> downsampled_points;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> downsampled_covs;
  downsampled_points.reserve(downsampling.voxels.size());
  downsampled_covs.reserve(downsampling.voxels.size());

  for (const auto& voxel : downsampling.voxels) {
    downsampled_points.push_back(voxel.second->mean);
    downsampled_covs.push_back(voxel.second->cov);
  }

  return std::make_shared<VoxelizedFrameCPU>(voxel_resolution, downsampled_points, downsampled_covs);
}

Frame::Ptr merge_frames_auto(
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& poses,
  const std::vector<Frame::ConstPtr>& frames,
  double downsample_resolution,
  double voxel_resolution) {
//
#ifdef BUILD_GTSAM_EXT_GPU
  if (frames[0]->points_gpu && frames[0]->covs_gpu) {
    return merge_frames_gpu(poses, frames, downsample_resolution, voxel_resolution);
  }
#endif

  return merge_frames(poses, frames, downsample_resolution, voxel_resolution);
}

double overlap(const GaussianVoxelMap::ConstPtr& target_, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta) {
  auto target = std::dynamic_pointer_cast<const GaussianVoxelMapCPU>(target_);
  if (target == nullptr) {
    std::cerr << "error: target CPU voxelmap has not been created!!" << std::endl;
    abort();
  }

  int num_overlap = 0;
  for (int i = 0; i < source->size(); i++) {
    Eigen::Vector4d pt = delta * source->points[i];
    Eigen::Vector3i coord = target->voxel_coord(pt);
    if (target->lookup_voxel(coord)) {
      num_overlap++;
    }
  }

  return static_cast<double>(num_overlap) / source->size();
}

double overlap(const Frame::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta) {
  return overlap(target->voxels, source, delta);
}

double overlap(
  const std::vector<GaussianVoxelMap::ConstPtr>& targets_,
  const Frame::ConstPtr& source,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas) {
  std::vector<GaussianVoxelMapCPU::ConstPtr> targets(targets_.size());
  for (int i = 0; i < targets_.size(); i++) {
    targets[i] = std::dynamic_pointer_cast<const GaussianVoxelMapCPU>(targets_[i]);
    if (!targets[i]) {
      std::cerr << "error: target CPU voxelmap has not been created!!" << std::endl;
    }
  }

  int num_overlap = 0;
  for (int i = 0; i < source->size(); i++) {
    for (int j = 0; j < targets.size(); j++) {
      const auto& target = targets[j];
      const auto& delta = deltas[j];

      Eigen::Vector4d pt = delta * source->points[i];
      Eigen::Vector3i coord = target->voxel_coord(pt);
      if (target->lookup_voxel(coord)) {
        num_overlap++;
        break;
      }
    }
  }

  return static_cast<double>(num_overlap) / source->size();
}

double overlap(
  const std::vector<Frame::ConstPtr>& targets_,
  const Frame::ConstPtr& source,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas) {
  std::vector<GaussianVoxelMap::ConstPtr> targets(targets_.size());
  for (int i = 0; i < targets_.size(); i++) {
    targets[i] = targets_[i]->voxels;
  }

  return overlap(targets, source, deltas);
}

double overlap_auto(const GaussianVoxelMap::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta) {
#ifdef BUILD_GTSAM_EXT_GPU
  if (source->points_gpu && std::dynamic_pointer_cast<const GaussianVoxelMapGPU>(target)) {
    return overlap_gpu(target, source, delta);
  }
#endif
  return overlap(target, source, delta);
}

double overlap_auto(
  const std::vector<GaussianVoxelMap::ConstPtr>& targets,
  const Frame::ConstPtr& source,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas) {
#ifdef BUILD_GTSAM_EXT_GPU
  if (source->points_gpu && !targets.empty() && std::dynamic_pointer_cast<const GaussianVoxelMapGPU>(targets[0])) {
    return overlap_gpu(targets, source, deltas);
  }
#endif
  return overlap(targets, source, deltas);
}

double overlap_auto(const Frame::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta) {
#ifdef BUILD_GTSAM_EXT_GPU
  if (source->points_gpu && target->voxels_gpu) {
    return overlap_gpu(target, source, delta);
  }
#endif
  return overlap(target, source, delta);
}

double overlap_auto(
  const std::vector<Frame::ConstPtr>& targets,
  const Frame::ConstPtr& source,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas) {
#ifdef BUILD_GTSAM_EXT_GPU
  if (source->points_gpu && !targets.empty() && targets[0]->voxels_gpu) {
    return overlap_gpu(targets, source, deltas);
  }
#endif
  return overlap(targets, source, deltas);
}

}  // namespace gtsam_ext
