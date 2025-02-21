// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <vector>
#include <Eigen/Core>
#include <gtsam_points/types/point_cloud.hpp>

namespace gtsam_points {

/**
 * @brief Covariance estimation parameters
 */
struct CovarianceEstimationParams {
public:
  enum RegularizationMethod { NONE, EIG };

  CovarianceEstimationParams() : num_threads(1), k_neighbors(10), regularization_method(EIG), eigen_values(1e-3, 1.0, 1.0) {}

public:
  int num_threads;                             ///< Number of threads
  int k_neighbors;                             ///< Number of neighboring points used for covariance estimation
  RegularizationMethod regularization_method;  ///< Regularization method
  Eigen::Vector3d eigen_values;                ///< Eigenvalues used for EIG regularization
};

/**
 * @brief Estimate point covariances from neighboring points
 * @param points       Input points
 * @param num_points   Number of input points
 * @param params       Estimation params
 * @return             Estimated covariances
 */
std::vector<Eigen::Matrix4d> estimate_covariances(const Eigen::Vector4d* points, int num_points, const CovarianceEstimationParams& params);

/**
 * @brief Estimate point covariances from neighboring points
 * @param points       Input points
 * @param num_points   Number of input points
 * @param k_neighbors  Number of neighboring points for covariance estimation
 * @param eigen_values Eigenvalues used for regularization (default=[1e-3, 1, 1])
 * @param num_threads  Number of threads
 * @return             Estimated covariances
 */
std::vector<Eigen::Matrix4d>
estimate_covariances(const Eigen::Vector4d* points, int num_points, int k_neighbors, const Eigen::Vector3d& eigen_values, int num_threads);

std::vector<Eigen::Matrix4d> estimate_covariances(const Eigen::Vector4d* points, int num_points, int k_neighbors = 10, int num_threads = 1);

template <typename Alloc>
std::vector<Eigen::Matrix4d> estimate_covariances(const std::vector<Eigen::Vector4d, Alloc>& points, int k_neighbors = 10, int num_threads = 1) {
  return estimate_covariances(points.data(), points.size(), k_neighbors, num_threads);
}

std::vector<Eigen::Matrix4d> estimate_covariances(const PointCloud& points, int k_neighbors = 10, int num_threads = 1);

}  // namespace gtsam_points