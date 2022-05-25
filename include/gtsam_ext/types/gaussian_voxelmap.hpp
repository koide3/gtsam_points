// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam_ext/types/frame.hpp>

namespace gtsam_ext {

/**
 * @brief Gaussian distribution voxelmap
 */
class GaussianVoxelMap {
public:
  using Ptr = std::shared_ptr<GaussianVoxelMap>;
  using ConstPtr = std::shared_ptr<const GaussianVoxelMap>;

  GaussianVoxelMap() {}
  virtual ~GaussianVoxelMap() {}

  /// Voxel resolution
  virtual double voxel_resolution() const = 0;

  /// Insert a point cloud frame into the voxelmap
  virtual void insert(const Frame& frame) = 0;
};

}  // namespace gtsam_ext