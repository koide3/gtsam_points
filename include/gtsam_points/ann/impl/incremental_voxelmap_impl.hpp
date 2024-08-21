// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <gtsam_points/ann/incremental_voxelmap.hpp>

#include <gtsam_points/ann/knn_result.hpp>
#include <gtsam_points/util/fast_floor.hpp>

namespace gtsam_points {

template <typename VoxelContents>
IncrementalVoxelMap<VoxelContents>::IncrementalVoxelMap(double leaf_size)
: inv_leaf_size(1.0 / leaf_size),
  lru_horizon(10),
  lru_clear_cycle(10),
  lru_counter(0),
  offsets(neighbor_offsets(7)) {}

template <typename VoxelContents>
IncrementalVoxelMap<VoxelContents>::~IncrementalVoxelMap() {}

template <typename VoxelContents>
void IncrementalVoxelMap<VoxelContents>::clear() {
  lru_counter = 0;
  flat_voxels.clear();
  voxels.clear();
}

template <typename VoxelContents>
void IncrementalVoxelMap<VoxelContents>::insert(const PointCloud& points) {
  // Insert points to the voxelmap
  for (size_t i = 0; i < points.size(); i++) {
    const Eigen::Vector3i coord = fast_floor(points.points[i] * inv_leaf_size).template head<3>();

    auto found = voxels.find(coord);
    if (found == voxels.end()) {
      auto voxel = std::make_shared<std::pair<VoxelInfo, VoxelContents>>(VoxelInfo(coord, lru_counter), VoxelContents());

      found = voxels.emplace_hint(found, coord, flat_voxels.size());
      flat_voxels.emplace_back(voxel);
    }

    auto& [info, voxel] = *flat_voxels[found->second];
    info.lru = lru_counter;
    voxel.add(voxel_setting, points, i);
  }

  if ((++lru_counter) % lru_clear_cycle == 0) {
    // Remove least recently used voxels
    auto remove_counter =
      std::remove_if(flat_voxels.begin(), flat_voxels.end(), [&](const std::shared_ptr<std::pair<VoxelInfo, VoxelContents>>& voxel) {
        return voxel->first.lru + lru_horizon < lru_counter;
      });
    flat_voxels.erase(remove_counter, flat_voxels.end());

    // Rehash
    voxels.clear();
    for (size_t i = 0; i < flat_voxels.size(); i++) {
      voxels[flat_voxels[i]->first.coord] = i;
    }
  }

  // Finalize voxel means and covs
  for (auto& voxel : flat_voxels) {
    voxel->second.finalize();
  }
}

template <typename VoxelContents>
size_t IncrementalVoxelMap<VoxelContents>::knn_search(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists, double max_sq_dist) const {
  const Eigen::Vector4d query = (Eigen::Vector4d() << pt[0], pt[1], pt[2], 1.0).finished();
  const Eigen::Vector3i center = fast_floor(query * inv_leaf_size).template head<3>();

  size_t voxel_index = 0;
  const auto index_transform = [&](const size_t point_index) { return calc_index(voxel_index, point_index); };

  KnnResult<-1, decltype(index_transform)> result(k_indices, k_sq_dists, k, index_transform, max_sq_dist);
  for (const auto& offset : offsets) {
    const Eigen::Vector3i coord = center + offset;
    const auto found = voxels.find(coord);
    if (found == voxels.end()) {
      continue;
    }

    voxel_index = found->second;
    const auto& voxel = flat_voxels[voxel_index]->second;
    voxel.knn_search(query, result);
  }

  return result.num_found();
}

template <typename VoxelContents>
std::vector<Eigen::Vector3i> IncrementalVoxelMap<VoxelContents>::neighbor_offsets(const int neighbor_voxel_mode) const {
  switch (neighbor_voxel_mode) {
    case 1:
      return std::vector<Eigen::Vector3i>{Eigen::Vector3i(0, 0, 0)};
    case 7:
      return std::vector<Eigen::Vector3i>{
        Eigen::Vector3i(0, 0, 0),
        Eigen::Vector3i(1, 0, 0),
        Eigen::Vector3i(-1, 0, 0),
        Eigen::Vector3i(0, 1, 0),
        Eigen::Vector3i(0, -1, 0),
        Eigen::Vector3i(0, 0, 1),
        Eigen::Vector3i(0, 0, -1)};
    case 19: {
      std::vector<Eigen::Vector3i> offsets;
      for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
          for (int k = -1; k <= 1; k++) {
            if (std::abs(i) == 1 && std::abs(j) == 1 && std::abs(k) == 1) {
              continue;
            }

            offsets.push_back(Eigen::Vector3i(i, j, k));
          }
        }
      }
      return offsets;
    }
    case 27: {
      std::vector<Eigen::Vector3i> offsets;
      for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
          for (int k = -1; k <= 1; k++) {
            offsets.push_back(Eigen::Vector3i(i, j, k));
          }
        }
      }
      return offsets;
    }

    default:
      std::cerr << "error: invalid neighbor voxel mode " << neighbor_voxel_mode << std::endl;
      std::cerr << "     : neighbor voxel mode must be 1, 7, 19, or 27" << std::endl;
      return std::vector<Eigen::Vector3i>();
  }
}

template <typename VoxelContents>
bool IncrementalVoxelMap<VoxelContents>::has_points() const {
  return flat_voxels.empty() ? false : frame::has_points(flat_voxels.front()->second);
}

template <typename VoxelContents>
bool IncrementalVoxelMap<VoxelContents>::has_normals() const {
  return flat_voxels.empty() ? false : frame::has_normals(flat_voxels.front()->second);
}

template <typename VoxelContents>
bool IncrementalVoxelMap<VoxelContents>::has_covs() const {
  return flat_voxels.empty() ? false : frame::has_covs(flat_voxels.front()->second);
}

template <typename VoxelContents>
bool IncrementalVoxelMap<VoxelContents>::has_intensities() const {
  return flat_voxels.empty() ? false : frame::has_intensities(flat_voxels.front()->second);
}

template <typename VoxelContents>
std::vector<Eigen::Vector4d> IncrementalVoxelMap<VoxelContents>::voxel_points() const {
  std::vector<Eigen::Vector4d> points;
  points.reserve(flat_voxels.size() * 10);
  visit_points([&](const auto& voxel, const int i) { points.emplace_back(frame::point(voxel, i)); });
  return points;
}

template <typename VoxelContents>
std::vector<Eigen::Vector4d> IncrementalVoxelMap<VoxelContents>::voxel_normals() const {
  std::vector<Eigen::Vector4d> normals;
  normals.reserve(flat_voxels.size() * 10);
  visit_points([&](const auto& voxel, const int i) { normals.emplace_back(frame::normal(voxel, i)); });
  return normals;
}

template <typename VoxelContents>
std::vector<Eigen::Matrix4d> IncrementalVoxelMap<VoxelContents>::voxel_covs() const {
  std::vector<Eigen::Matrix4d> covs;
  covs.reserve(flat_voxels.size() * 10);
  visit_points([&](const auto& voxel, const int i) { covs.emplace_back(frame::cov(voxel, i)); });
  return covs;
}

template <typename VoxelContents>
std::vector<double> IncrementalVoxelMap<VoxelContents>::voxel_intensities() const {
  std::vector<double> intensities;
  intensities.reserve(flat_voxels.size() * 10);
  visit_points([&](const auto& voxel, const int i) { intensities.emplace_back(frame::intensity(voxel, i)); });
  return intensities;
}

template <typename VoxelContents>
PointCloudCPU::Ptr IncrementalVoxelMap<VoxelContents>::voxel_data() const {
  auto frame = std::make_shared<PointCloudCPU>();
  frame->points_storage.reserve(flat_voxels.size() * 10);
  if (has_normals()) {
    frame->normals_storage.reserve(flat_voxels.size() * 10);
  }
  if (has_covs()) {
    frame->covs_storage.reserve(flat_voxels.size() * 10);
  }
  if (has_intensities()) {
    frame->intensities_storage.reserve(flat_voxels.size() * 10);
  }

  visit_points([&](const auto& voxel, const int i) {
    frame->points_storage.emplace_back(frame::point(voxel, i));
    if (frame::has_normals(voxel)) {
      frame->normals_storage.emplace_back(frame::normal(voxel, i));
    }
    if (frame::has_covs(voxel)) {
      frame->covs_storage.emplace_back(frame::cov(voxel, i));
    }
    if (frame::has_intensities(voxel)) {
      frame->intensities_storage.emplace_back(frame::intensity(voxel, i));
    }
  });

  frame->num_points = frame->points_storage.size();
  frame->points = frame->points_storage.data();
  frame->normals = frame->normals_storage.empty() ? nullptr : frame->normals_storage.data();
  frame->covs = frame->covs_storage.empty() ? nullptr : frame->covs_storage.data();
  frame->intensities = frame->intensities_storage.empty() ? nullptr : frame->intensities_storage.data();

  return frame;
}

}  // namespace gtsam_points
