// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/util/covariance_estimation.hpp>

#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <nanoflann.hpp>
#include <gtsam_points/ann/kdtree.hpp>

namespace gtsam_points {

std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
estimate_covariances(const Eigen::Vector4d* points, int num_points, const CovarianceEstimationParams& params) {
  KdTree tree(points, num_points);
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> covs(num_points);

#pragma omp parallel for num_threads(params.num_threads) schedule(guided, 8)
  for (int i = 0; i < num_points; i++) {
    std::vector<size_t> k_indices(params.k_neighbors);
    std::vector<double> k_sq_dists(params.k_neighbors);
    tree.knn_search(points[i].data(), params.k_neighbors, &k_indices[0], &k_sq_dists[0]);

    Eigen::Vector4d sum_points = Eigen::Vector4d::Zero();
    Eigen::Matrix4d sum_covs = Eigen::Matrix4d::Zero();

    for (int j = 0; j < params.k_neighbors; j++) {
      const auto& pt = points[k_indices[j]];
      sum_points += pt;
      sum_covs += pt * pt.transpose();
    }

    Eigen::Vector4d mean = sum_points / params.k_neighbors;
    Eigen::Matrix4d cov = (sum_covs - mean * sum_points.transpose()) / params.k_neighbors;

    switch (params.regularization_method) {
      default:
        break;
      case CovarianceEstimationParams::EIG: {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig;
        eig.computeDirect(cov.block<3, 3>(0, 0));

        covs[i].setZero();
        covs[i].block<3, 3>(0, 0) = eig.eigenvectors() * params.eigen_values.asDiagonal() * eig.eigenvectors().inverse();
      } break;
    }
  }

  return covs;
}

std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
estimate_covariances(const Eigen::Vector4d* points, int num_points, int k_neighbors, int num_threads) {
  CovarianceEstimationParams params;
  params.k_neighbors = k_neighbors;
  params.num_threads = num_threads;
  return estimate_covariances(points, num_points, params);
}

std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
estimate_covariances(const Eigen::Vector4d* points, int num_points, int k_neighbors, const Eigen::Vector3d& eigen_values, int num_threads) {
  CovarianceEstimationParams params;
  params.k_neighbors = k_neighbors;
  params.eigen_values = eigen_values;
  params.num_threads = num_threads;
  return estimate_covariances(points, num_points, params);
}

}  // namespace gtsam_points
