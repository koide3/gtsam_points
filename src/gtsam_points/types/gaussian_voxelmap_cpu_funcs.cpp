// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>

#include <memory>
#include <fstream>
#include <iostream>
#include <unordered_set>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam_points/config.hpp>
#include <gtsam_points/util/fast_floor.hpp>
#include <gtsam_points/util/reservoir.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>

#ifdef GTSAM_POINTS_USE_CUDA
#include <gtsam_points/types/gaussian_voxelmap_gpu.hpp>
#endif

namespace gtsam_points {

// merge_frames
PointCloud::Ptr
merge_frames(const std::vector<Eigen::Isometry3d>& poses, const std::vector<PointCloud::ConstPtr>& frames, double downsample_resolution) {
  return merge_frames(poses, frames, downsample_resolution, std::numeric_limits<size_t>::max());
}

PointCloud::Ptr merge_frames(
  const std::vector<Eigen::Isometry3d>& poses,
  const std::vector<PointCloud::ConstPtr>& frames,
  double downsample_resolution,
  size_t max_num_points) {
  constexpr int coord_bits = 21;
  constexpr std::uint64_t coord_bitmask = (1ull << coord_bits) - 1;
  const int coord_offset = 1 << (coord_bits - 1);  // Coordinate offset to make values positive

  constexpr int frame_id_bits = 32;
  constexpr std::uint64_t frame_id_bitmask = (1ull << frame_id_bits) - 1;

  const double inv_resolution = 1.0 / downsample_resolution;
  const size_t num_all_points = std::accumulate(frames.begin(), frames.end(), 0, [](size_t sum, const auto& points) { return points->size() + sum; });

  std::vector<std::pair<std::uint64_t, std::uint64_t>> coords_indices;
  coords_indices.reserve(num_all_points);

  for (size_t frame_id = 0; frame_id < frames.size(); frame_id++) {
    const auto& frame = frames[frame_id];
    const Eigen::Isometry3d pose = poses.front().inverse() * poses[frame_id];

    for (size_t point_id = 0; point_id < frame->size(); point_id++) {
      const Eigen::Vector4d point = pose * frame->points[point_id];
      const Eigen::Array4i coord = fast_floor(point * inv_resolution) + coord_offset;
      if ((coord > coord_bitmask).any()) {
        std::cerr << "warning: out of valid voxel range!! " << std::endl;
        continue;
      }

      const std::uint64_t index = (frame_id << frame_id_bits) | (point_id & frame_id_bitmask);
      coords_indices.emplace_back(
        (static_cast<std::uint64_t>(coord.x() & coord_bitmask) << (coord_bits * 0)) |
          (static_cast<std::uint64_t>(coord.y() & coord_bitmask) << (coord_bits * 1)) |
          (static_cast<std::uint64_t>(coord.z() & coord_bitmask) << (coord_bits * 2)),
        index);
    }
  }

  std::sort(coords_indices.begin(), coords_indices.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

  std::vector<std::uint64_t> sampled_indices;
  sampled_indices.reserve(std::min<size_t>(max_num_points, coords_indices.size() / 10));

  std::mt19937 mt;
  Reservoir<std::uint64_t, 1> reservoir;

  for (int i = 0; i < coords_indices.size(); i++) {
    if (i && coords_indices[i - 1].first != coords_indices[i].first) {
      sampled_indices.emplace_back(reservoir.front());
      reservoir = Reservoir<std::uint64_t, 1>();
    }

    reservoir.push(coords_indices[i].second, mt);
  }
  sampled_indices.emplace_back(reservoir.front());

  if (sampled_indices.size() > max_num_points) {
    std::vector<std::uint64_t> subsampled_indices(max_num_points);
    std::sample(sampled_indices.begin(), sampled_indices.end(), subsampled_indices.begin(), max_num_points, mt);
    sampled_indices = std::move(subsampled_indices);
  }

  auto merged = sample_multi_frames(frames, sampled_indices);

  for (size_t i = 0; i < merged->size(); i++) {
    const int frame_id = (sampled_indices[i] >> frame_id_bits) & frame_id_bitmask;
    const auto& pose = poses[frame_id];

    merged->points[i] = pose * merged->points[i];

    if (merged->covs) {
      merged->covs[i] = pose.matrix() * merged->covs[i] * pose.matrix().transpose();
    }

    if (merged->normals) {
      merged->normals[i] = pose.matrix() * merged->normals[i];
    }
  }

  return merged;
}

PointCloud::Ptr
merge_frames_auto(const std::vector<Eigen::Isometry3d>& poses, const std::vector<PointCloud::ConstPtr>& frames, double downsample_resolution) {
  // TODO: Implement GPU version of merge_frames and call it here when the input frames are on GPU
  // #ifdef GTSAM_POINTS_USE_CUDA
  //   if (frames[0]->points_gpu && frames[0]->covs_gpu) {
  //     return merge_frames_gpu(poses, frames, downsample_resolution);
  //   }
  // #endif

  return merge_frames(poses, frames, downsample_resolution);
}

double overlap(const GaussianVoxelMap::ConstPtr& target_, const PointCloud::ConstPtr& source, const Eigen::Isometry3d& T_target_source) {
  auto target = std::dynamic_pointer_cast<const GaussianVoxelMapCPU>(target_);
  if (target == nullptr) {
    std::cerr << "error: target CPU voxelmap has not been created!!" << std::endl;
    abort();
  }

  int num_overlap = 0;
  for (int i = 0; i < source->size(); i++) {
    Eigen::Vector4d pt = T_target_source * source->points[i];
    Eigen::Vector3i coord = target->voxel_coord(pt);
    if (target->lookup_voxel_index(coord) >= 0) {
      num_overlap++;
    }
  }

  return static_cast<double>(num_overlap) / source->size();
}

double overlap(
  const std::vector<GaussianVoxelMap::ConstPtr>& targets_,
  const PointCloud::ConstPtr& source,
  const std::vector<Eigen::Isometry3d>& Ts_target_source) {
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
      const auto& T_target_source = Ts_target_source[j];

      Eigen::Vector4d pt = T_target_source * source->points[i];
      Eigen::Vector3i coord = target->voxel_coord(pt);
      if (target->lookup_voxel_index(coord) >= 0) {
        num_overlap++;
        break;
      }
    }
  }

  return static_cast<double>(num_overlap) / source->size();
}

double overlap_auto(const GaussianVoxelMap::ConstPtr& target, const PointCloud::ConstPtr& source, const Eigen::Isometry3d& delta) {
#ifdef GTSAM_POINTS_USE_CUDA
  if (source->points_gpu && std::dynamic_pointer_cast<const GaussianVoxelMapGPU>(target)) {
    return overlap_gpu(target, source, delta);
  }
#endif
  return overlap(target, source, delta);
}

double overlap_auto(
  const std::vector<GaussianVoxelMap::ConstPtr>& targets,
  const PointCloud::ConstPtr& source,
  const std::vector<Eigen::Isometry3d>& deltas) {
#ifdef GTSAM_POINTS_USE_CUDA
  if (source->points_gpu && !targets.empty() && std::dynamic_pointer_cast<const GaussianVoxelMapGPU>(targets[0])) {
    return overlap_gpu(targets, source, deltas);
  }
#endif
  return overlap(targets, source, deltas);
}

}  // namespace gtsam_points
