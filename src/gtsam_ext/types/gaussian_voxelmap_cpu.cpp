// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>

#include <memory>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/functional/hash.hpp>

namespace gtsam_ext {

size_t Vector3iHash::operator()(const Eigen::Vector3i& x) const {
  size_t seed = 0;
  boost::hash_combine(seed, x[0]);
  boost::hash_combine(seed, x[1]);
  boost::hash_combine(seed, x[2]);
  return seed;
}

GaussianVoxel::GaussianVoxel() {
  num_points = 0;
  mean.setZero();
  cov.setZero();
}

void GaussianVoxel::append(const Eigen::Vector4d& mean_, const Eigen::Matrix4d& cov_) {
  num_points++;
  mean += mean_;
  cov += cov_;
}

void GaussianVoxel::finalize() {
  mean /= num_points;
  cov /= num_points;
}

GaussianVoxelMapCPU::GaussianVoxelMapCPU(double resolution) : resolution(resolution) {}

GaussianVoxelMapCPU::~GaussianVoxelMapCPU() {}

Eigen::Vector3i GaussianVoxelMapCPU::voxel_coord(const Eigen::Vector4d& x) const {
  return (x.array() / resolution - 0.5).floor().cast<int>().head<3>();
}

GaussianVoxel::Ptr GaussianVoxelMapCPU::lookup_voxel(const Eigen::Vector3i& coord) const {
  auto found = voxels.find(coord);
  if (found == voxels.end()) {
    return nullptr;
  }

  return found->second;
}

void GaussianVoxelMapCPU::create_voxelmap(const Frame& frame) {
  if (!frame.points || !frame.covs) {
    std::cerr << "error: points/covs not allocated!!" << std::endl;
    abort();
  }

  voxels.clear();

  for (int i = 0; i < frame.num_points; i++) {
    Eigen::Vector3i coord = voxel_coord(frame.points[i]);

    auto found = voxels.find(coord);
    if (found == voxels.end()) {
      GaussianVoxel::Ptr voxel(new GaussianVoxel());
      found = voxels.insert(found, std::make_pair(coord, voxel));
    }

    auto& voxel = found->second;
    voxel->append(frame.points[i], frame.covs[i]);
  }

  for (auto& voxel : voxels) {
    voxel.second->finalize();
  }
}

}  // namespace gtsam_ext
