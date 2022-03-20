// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_ext/ann/ivox.hpp>

#include <iostream>

namespace gtsam_ext {

size_t XORVector3iHash::operator()(const Eigen::Vector3i& x) const {
  const size_t p1 = 73856093;
  const size_t p2 = 19349669;
  const size_t p3 = 83492791;
  return static_cast<size_t>((x[0] * p1) ^ (x[1] * p2) ^ (x[2] * p3));
}

LinearContainer::LinearContainer(const int lru_count) : last_lru_count(lru_count), serial_id(0) {}

LinearContainer::~LinearContainer() {}

void LinearContainer::insert(const Eigen::Vector4d& point, const double insertion_dist_sq_thresh) {
  double min_dist = std::numeric_limits<double>::max();
  for (const auto& p : points) {
    min_dist = std::min(min_dist, (p - point).squaredNorm());
  }

  if (min_dist > insertion_dist_sq_thresh) {
    points.push_back(point);
  }
}

void LinearContainer::insert(const Frame& frame, const int i, const double insertion_dist_sq_thresh) {
  const auto& point = frame.points[i];

  double min_dist = std::numeric_limits<double>::max();
  for (const auto& p : points) {
    min_dist = std::min(min_dist, (p - point).squaredNorm());
  }

  if (min_dist > insertion_dist_sq_thresh) {
    points.push_back(point);

    if (frame.has_normals()) {
      normals.push_back(frame.normals[i]);
    }
    if (frame.has_covs()) {
      covs.push_back(frame.covs[i]);
    }
    if (frame.has_intensities()) {
      intensities.push_back(frame.intensities[i]);
    }
  }
}

iVox::iVox(const double voxel_resolution) : voxel_resolution(voxel_resolution) {
  insertion_dist_sq_thresh = std::pow(0.05, 2);
  lru_thresh = 10;
  lru_count = 0;

  neighbor_voxel_mode = 7;

  has_points = false;
  has_normals = false;
  has_covs = false;
  has_intensities = false;
}

iVox::~iVox() {}

const Eigen::Vector3i iVox::voxel_coord(const Eigen::Vector4d& point) const {
  const Eigen::Vector4i coord = (point / voxel_resolution).array().floor().cast<int>();
  return coord.head<3>();
}

std::vector<Eigen::Vector3i> iVox::neighbor_offsets() const {
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

void iVox::insert(const Eigen::Vector4d* points, int num_points) {
  lru_count++;

  for (int i = 0; i < num_points; i++) {
    const auto& point = points[i];
    const Eigen::Vector3i coord = voxel_coord(point);

    auto found = voxelmap.find(coord);
    if (found == voxelmap.end()) {
      auto new_voxel = std::make_shared<LinearContainer>(lru_count);
      found = voxelmap.insert(found, std::make_pair(coord, new_voxel));
    }

    if (found->second->size() >= (1 << point_id_bits) - 1) {
      std::cerr << "warning: too many points in voxel!!" << std::endl;
      std::cerr << "       : skip point insertion!!" << std::endl;
      continue;
    }

    found->second->last_lru_count = lru_count;
    found->second->insert(point, insertion_dist_sq_thresh);
  }

  voxels.clear();
  for (auto& voxel : voxelmap) {
    if (voxels.size() >= (1 << voxel_id_bits) - 1) {
      std::cerr << "warning: too many voxels!!" << std::endl;
      continue;
    }

    voxel.second->serial_id = voxels.size();
    voxels.push_back(voxel.second);
  }
}

void iVox::insert(const Frame& frame) {
  // Attribute check
  if (!has_points) {
    has_points = frame.has_points();
    has_normals = frame.has_normals();
    has_covs = frame.has_covs();
    has_intensities = frame.has_intensities();
  } else {
    if (has_normals != frame.has_normals()) {
      std::cerr << "error: inconsistent point attribute (normal)" << std::endl;
    }
    if (has_covs != frame.has_covs()) {
      std::cerr << "error: inconsistent point attribute (cov)" << std::endl;
    }
    if (has_intensities != frame.has_intensities()) {
      std::cerr << "error: inconsistent point attribute (intensity)" << std::endl;
    }
  }

  lru_count++;

  // Insert points
  for (int i = 0; i < frame.size(); i++) {
    const auto& point = frame.points[i];
    const Eigen::Vector3i coord = voxel_coord(point);

    auto found = voxelmap.find(coord);
    if (found == voxelmap.end()) {
      auto new_voxel = std::make_shared<LinearContainer>(lru_count);
      found = voxelmap.insert(found, std::make_pair(coord, new_voxel));
    }

    if (found->second->size() >= (1 << point_id_bits) - 1) {
      std::cerr << "warning: too many points in voxel!!" << std::endl;
      std::cerr << "       : skip point insertion!!" << std::endl;
      continue;
    }

    found->second->last_lru_count = lru_count;
    found->second->insert(frame, i, insertion_dist_sq_thresh);
  }

  // Remove voxels that are not used recently
  const int lru_horizon = lru_count - lru_thresh;
  for (auto voxel = voxelmap.begin(); voxel != voxelmap.end(); voxel++) {
    if (voxel->second->last_lru_count < lru_horizon) {
      voxel = voxelmap.erase(voxel);
    }
  }

  // Create lat voxel list
  voxels.clear();
  for (auto& voxel : voxelmap) {
    if (voxels.size() >= (1 << voxel_id_bits) - 1) {
      std::cerr << "warning: too many voxels!!" << std::endl;
      continue;
    }

    voxel.second->serial_id = voxels.size();
    voxels.push_back(voxel.second);
  }
}

size_t iVox::knn_search(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists) const {
  const Eigen::Vector4d point(pt[0], pt[1], pt[2], 1.0);
  const Eigen::Vector3i center = voxel_coord(point);

  const std::vector<Eigen::Vector3i> offsets = neighbor_offsets();

  // Find neighbor points
  std::vector<std::pair<size_t, double>> neighbors;
  for (const auto& offset : offsets) {
    const Eigen::Vector3i coord = center + offset;
    const auto found = voxelmap.find(coord);

    if (found == voxelmap.end()) {
      continue;
    }

    found->second->last_lru_count = lru_count;

    for (int i = 0; i < found->second->size(); i++) {
      const size_t point_id = (found->second->serial_id << point_id_bits) | i;
      const double dist = (point - found->second->points[i]).squaredNorm();
      neighbors.push_back(std::make_pair(point_id, dist));
    }
  }

  // Sort results and return them
  if (neighbors.size() <= k) {
    std::sort(neighbors.begin(), neighbors.end(), [](const std::pair<size_t, double>& lhs, const std::pair<size_t, double>& rhs) {
      return lhs.second < rhs.second;
    });
    for (int i = 0; i < neighbors.size(); i++) {
      k_indices[i] = neighbors[i].first;
      k_sq_dists[i] = neighbors[i].second;
    }
    return neighbors.size();
  }

  std::partial_sort(
    neighbors.begin(),
    neighbors.begin() + k,
    neighbors.end(),
    [](const std::pair<size_t, double>& lhs, const std::pair<size_t, double>& rhs) { return lhs.second < rhs.second; });

  for (int i = 0; i < k; i++) {
    k_indices[i] = neighbors[i].first;
    k_sq_dists[i] = neighbors[i].second;
  }
  return k;
}

const Eigen::Vector4d& iVox::point(const size_t i) const {
  const size_t voxel_id = i >> point_id_bits;
  const size_t point_id = i & ((1 << point_id_bits) - 1);
  return voxels[voxel_id]->points[point_id];
}

const Eigen::Vector4d& iVox::normal(const size_t i) const {
  const size_t voxel_id = i >> point_id_bits;
  const size_t point_id = i & ((1 << point_id_bits) - 1);
  return voxels[voxel_id]->normals[point_id];
}

const Eigen::Matrix4d& iVox::cov(const size_t i) const {
  const size_t voxel_id = i >> point_id_bits;
  const size_t point_id = i & ((1 << point_id_bits) - 1);
  return voxels[voxel_id]->covs[point_id];
}

double iVox::intensity(const size_t i) const {
  const size_t voxel_id = i >> point_id_bits;
  const size_t point_id = i & ((1 << point_id_bits) - 1);
  return voxels[voxel_id]->intensities[point_id];
}

}  // namespace gtsam_ext