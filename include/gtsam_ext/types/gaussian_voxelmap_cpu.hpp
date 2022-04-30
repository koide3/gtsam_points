// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <atomic>
#include <memory>
#include <unordered_map>

#include <Eigen/Core>
#include <gtsam_ext/util/vector3i_hash.hpp>
#include <gtsam_ext/types/gaussian_voxelmap.hpp>

namespace gtsam_ext {

struct GaussianVoxel {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<GaussianVoxel>;
  using ConstPtr = std::shared_ptr<const GaussianVoxel>;

  GaussianVoxel();

  void append(const Eigen::Vector4d& mean_, const Eigen::Matrix4d& cov_);
  void finalize();

public:
  bool finalized;
  mutable std::atomic_int last_lru_count;

  int num_points;
  Eigen::Vector4d mean;
  Eigen::Matrix4d cov;
};

class GaussianVoxelMapCPU : public GaussianVoxelMap {
public:
  using Ptr = std::shared_ptr<GaussianVoxelMapCPU>;
  using ConstPtr = std::shared_ptr<const GaussianVoxelMapCPU>;

  GaussianVoxelMapCPU(double resolution);
  virtual ~GaussianVoxelMapCPU();

  virtual double voxel_resolution() const override { return resolution; }
  virtual void insert(const Frame& frame) override;

  Eigen::Vector3i voxel_coord(const Eigen::Vector4d& x) const;
  GaussianVoxel::Ptr lookup_voxel(const Eigen::Vector3i& coord) const;

public:
  using VoxelMap = std::unordered_map<
    Eigen::Vector3i,
    GaussianVoxel::Ptr,
    Vector3iHash,
    std::equal_to<Eigen::Vector3i>,
    Eigen::aligned_allocator<std::pair<const Eigen::Vector3i, GaussianVoxel::Ptr>>>;

  int lru_count;
  int lru_thresh;

  double resolution;
  VoxelMap voxels;
};

}  // namespace gtsam_ext
