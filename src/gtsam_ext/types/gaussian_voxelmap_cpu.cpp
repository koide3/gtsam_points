// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>

#include <memory>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace gtsam_ext {

GaussianVoxel::GaussianVoxel() {
  last_lru_count = 0;
  finalized = false;
  num_points = 0;
  mean.setZero();
  cov.setZero();
}

void GaussianVoxel::append(const Eigen::Vector4d& mean_, const Eigen::Matrix4d& cov_) {
  if (finalized) {
    mean = num_points * mean;
    cov = num_points * cov;
    finalized = false;
  }

  num_points++;
  mean += mean_;
  cov += cov_;
}

void GaussianVoxel::finalize() {
  if (!finalized) {
    mean /= num_points;
    cov /= num_points;
  }
  finalized = true;
}

GaussianVoxelMapCPU::GaussianVoxelMapCPU(double resolution) : lru_count(0), lru_thresh(10), resolution(resolution) {}

GaussianVoxelMapCPU::~GaussianVoxelMapCPU() {}

Eigen::Vector3i GaussianVoxelMapCPU::voxel_coord(const Eigen::Vector4d& x) const {
  return (x.array() / resolution - 0.5).floor().cast<int>().head<3>();
}

GaussianVoxel::Ptr GaussianVoxelMapCPU::lookup_voxel(const Eigen::Vector3i& coord) const {
  auto found = voxels.find(coord);
  if (found == voxels.end()) {
    return nullptr;
  }

  found->second->last_lru_count = lru_count;
  return found->second;
}

void GaussianVoxelMapCPU::insert(const Frame& frame) {
  if (!frame::has_points(frame) || !frame::has_covs(frame)) {
    std::cerr << "error: points/covs not allocated!!" << std::endl;
    abort();
  }

  lru_count++;

  for (int i = 0; i < frame::size(frame); i++) {
    Eigen::Vector3i coord = voxel_coord(frame::point(frame, i));

    auto found = voxels.find(coord);
    if (found == voxels.end()) {
      GaussianVoxel::Ptr voxel(new GaussianVoxel());
      found = voxels.insert(found, std::make_pair(coord, voxel));
    }

    auto& voxel = found->second;
    voxel->last_lru_count = lru_count;
    voxel->append(frame::point(frame, i), frame::cov(frame, i));
  }

  // Remove voxels that are not used recently
  const int lru_horizon = lru_count - lru_thresh;
  for (auto voxel = voxels.begin(); voxel != voxels.end();) {
    if (voxel->second->last_lru_count < lru_horizon) {
      voxel = voxels.erase(voxel);
    } else {
      voxel++;
    }
  }

  // Finalize voxel means and covs
  for (auto& voxel : voxels) {
    voxel.second->finalize();
  }
}

}  // namespace gtsam_ext
