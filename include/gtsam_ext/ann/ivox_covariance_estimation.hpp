// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam_ext/ann/ivox.hpp>

namespace gtsam_ext {

class iVoxCovarianceEstimation : public iVox {
public:
  iVoxCovarianceEstimation(
    const double voxel_resolution,
    const double min_points_dist,
    const int lru_thresh,
    const int k_neighbors,
    const int num_threads = 1);
  ~iVoxCovarianceEstimation() override;

  virtual void insert(const Frame& frame) override;

  const Eigen::Vector4d& normal(const size_t i) const;
  const Eigen::Matrix4d& cov(const size_t i) const;

  virtual std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> voxel_normals() const override;
  virtual std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> voxel_covs() const override;

private:
  std::pair<Eigen::Matrix4d, Eigen::Vector4d>
  estimate_cov_and_normal(const Eigen::Vector4d& point, const int num_found, const size_t* k_indices, const double* k_sq_dists) const;

private:
  int k_neighbors;
  int num_threads;

  VoxelMap covmap;
  std::vector<LinearContainer*> covs;
};

namespace frame {

template <>
struct traits<iVoxCovarianceEstimation> {
  static bool has_points(const iVoxCovarianceEstimation& ivox) { return ivox.has_points(); }
  static bool has_normals(const iVoxCovarianceEstimation& ivox) { return true; }
  static bool has_covs(const iVoxCovarianceEstimation& ivox) { return true; }

  static const Eigen::Vector4d& point(const iVoxCovarianceEstimation& ivox, size_t i) { return ivox.point(i); }
  static const Eigen::Vector4d& normal(const iVoxCovarianceEstimation& ivox, size_t i) { return ivox.normal(i); }
  static const Eigen::Matrix4d& cov(const iVoxCovarianceEstimation& ivox, size_t i) { return ivox.cov(i); }
};

}  // namespace frame

}  // namespace gtsam_ext