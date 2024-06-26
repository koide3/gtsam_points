// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam_points/ann/ivox.hpp>
#include <gtsam_points/factors/intensity_gradients.hpp>

namespace gtsam_points {

class IntensityGradientsiVox : public iVox {
public:
  IntensityGradientsiVox(const double voxel_resolution, const double min_points_dist, int k_neighbors, int num_threads = 1);
  virtual ~IntensityGradientsiVox() override;

  /**
   * @brief Insert points and all available attributes into the iVox
   * @param frame   Input frame
   */
  virtual void insert(const PointCloud& frame) override;

  const Eigen::Vector4d& normal(const size_t i) const;
  const Eigen::Vector4d& intensity_gradient(const size_t i) const;

  std::vector<Eigen::Vector4d> voxel_normals() const override;
  std::vector<double> voxel_intensities() const;
  std::vector<Eigen::Vector4d> voxel_intensity_gradients() const;

private:
  Eigen::Vector4d estimate_normal(const Eigen::Vector4d& point, const int num_found, const size_t* k_indices, const double* k_sq_dists) const;
  Eigen::Vector4d estimate_gradient(
    const Eigen::Vector4d& point,
    const Eigen::Vector4d& normal,
    const double intensity,
    const int num_found,
    const size_t* k_indices,
    const double* k_sq_dists) const;

private:
  int k_neighbors;
  int num_threads;

  using GradientsMap = std::unordered_map<Eigen::Vector3i, LinearContainer::Ptr, XORVector3iHash>;
  GradientsMap gradmap;                     // Voxelmap
  std::vector<LinearContainer::Ptr> grads;  // Flattened voxelmap for linear indexing
};

namespace frame {

template <>
struct traits<IntensityGradientsiVox> {
  static bool has_points(const IntensityGradientsiVox& ivox) { return ivox.has_points(); }
  static bool has_normals(const IntensityGradientsiVox& ivox) { return true; }
  static bool has_covs(const IntensityGradientsiVox& ivox) { return ivox.has_covs(); }
  static bool has_intensities(const IntensityGradientsiVox& ivox) { return ivox.has_intensities(); }

  static const Eigen::Vector4d& point(const IntensityGradientsiVox& ivox, size_t i) { return ivox.point(i); }
  static const Eigen::Vector4d& normal(const IntensityGradientsiVox& ivox, size_t i) { return ivox.normal(i); }
  static const Eigen::Matrix4d& cov(const IntensityGradientsiVox& ivox, size_t i) { return ivox.cov(i); }
  static double intensity(const IntensityGradientsiVox& ivox, size_t i) { return ivox.intensity(i); }
  static const Eigen::Vector4d& intensity_gradient(const IntensityGradientsiVox& ivox, size_t i) { return ivox.intensity_gradient(i); }
};

}  // namespace frame

}  // namespace gtsam_points