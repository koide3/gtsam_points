// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>

#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/types/frame_traits.hpp>

namespace gtsam_points {

/**
 * @brief Intensity gradients for colored ICP
 *        Park et al., "Colored Point Cloud Registration Revisited", ICCV2017
 */
class IntensityGradients {
public:
  using Ptr = std::shared_ptr<IntensityGradients>;
  using ConstPtr = std::shared_ptr<const IntensityGradients>;

  /**
   * @brief Estimate intensity gradients of points
   * @note  frame is required to have normals and intensities
   * @param frame Input frame
   * @param k_neighbors Number of neighbors used for intensity gradient estimation
   * @param num_threads Number of threads
   */
  static IntensityGradients::Ptr estimate(const PointCloud::ConstPtr& frame, int k_neighbors = 10, int num_threads = 1);

  /**
   * @brief Estimate normals, covs, and intensity gradients
   * @note  If frame doesn't have normals and covs, these attributes are also estimated alongside intensity gradients
   * @param frame [in/out] Input frame
   * @param k_geom_neighbors Number of neighbors used for normal and covariance estimation
   * @param k_photo_neighbors Number of neighbors used for intensity gradient estimation
   * @param num_threads Number of threads
   */
  static IntensityGradients::Ptr
  estimate(const gtsam_points::PointCloudCPU::Ptr& frame, int k_geom_neighbors = 10, int k_photo_neighbors = 20, int num_threads = 1);

  static IntensityGradients::Ptr estimate(const PointCloud::ConstPtr& frame, const std::vector<int>& neighbors, int k_photo_neighbors);

public:
  std::vector<Eigen::Vector4d> intensity_gradients;
};

namespace frame {

template <>
struct traits<IntensityGradients> {
  static const Eigen::Vector4d& intensity_gradient(const IntensityGradients& grads, size_t i) { return grads.intensity_gradients[i]; }
};

}  // namespace frame

}  // namespace gtsam_points