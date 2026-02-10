// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>

namespace gtsam_points {

/// @brief Compact Gaussian voxel data for IO
struct GaussianVoxelData {
public:
  GaussianVoxelData() {}

  GaussianVoxelData(const Eigen::Vector3i& coord, const GaussianVoxel& voxel) {
    const auto& mean = voxel.mean;
    const auto& cov = voxel.cov;
    const auto intensity = voxel.intensity;

    this->coord = coord;
    this->num_points = voxel.num_points;
    this->mean << mean[0], mean[1], mean[2];
    this->cov << cov(0, 0), cov(0, 1), cov(0, 2), cov(1, 1), cov(1, 2), cov(2, 2);
    this->intensity = intensity;
  }

  std::shared_ptr<std::pair<VoxelInfo, GaussianVoxel>> uncompact() const {
    auto uncompacted = std::make_shared<std::pair<VoxelInfo, GaussianVoxel>>();
    uncompacted->first.lru = 0;
    uncompacted->first.coord = coord;

    auto& voxel = uncompacted->second;
    voxel.finalized = true;
    voxel.num_points = num_points;
    voxel.mean << mean.cast<double>(), 1.0;

    voxel.cov(0, 0) = cov[0];
    voxel.cov(0, 1) = voxel.cov(1, 0) = cov[1];
    voxel.cov(0, 2) = voxel.cov(2, 0) = cov[2];
    voxel.cov(1, 1) = cov[3];
    voxel.cov(1, 2) = voxel.cov(2, 1) = cov[4];
    voxel.cov(2, 2) = cov[5];

    voxel.intensity = intensity;
    return uncompacted;
  }

public:
  Eigen::Vector3i coord;
  int num_points;
  Eigen::Vector3f mean;
  Eigen::Matrix<float, 6, 1> cov;
  float intensity;
};

}