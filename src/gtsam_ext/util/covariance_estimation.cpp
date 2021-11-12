// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_ext/util/covariance_estimation.hpp>

#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <nanoflann.hpp>
#include <gtsam_ext/ann/kdtree.hpp>

namespace gtsam_ext {

std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> estimate_covariances(const Eigen::Vector4d* points, int num_points, int k_neighbors) {
  //
  KdTree tree(points, num_points);

  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> covs(num_points);
  for (int i = 0; i < num_points; i++) {
    std::vector<size_t> k_indices(k_neighbors);
    std::vector<double> k_sq_dists(k_neighbors);
    tree.knn_search(points[i].data(), k_neighbors, &k_indices[0], &k_sq_dists[0]);

    Eigen::Vector4d sum_points = Eigen::Vector4d::Zero();
    Eigen::Matrix4d sum_covs = Eigen::Matrix4d::Zero();

    for (int j = 0; j < k_neighbors; j++) {
      const auto& pt = points[k_indices[j]];
      sum_points += pt;
      sum_covs += pt * pt.transpose();
    }

    Eigen::Vector4d mean = sum_points / k_neighbors;
    Eigen::Matrix4d cov = (sum_covs - mean * sum_points.transpose()) / k_neighbors;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig;
    eig.computeDirect(cov.block<3, 3>(0, 0));

    Eigen::Vector3d values(1e-3, 1.0, 1.0);

    covs[i].setZero();
    covs[i].block<3, 3>(0, 0) = eig.eigenvectors() * values.asDiagonal() * eig.eigenvectors().inverse();
  }

  return covs;
}

std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> estimate_covariances(
  const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
  int k_neighbors) {
  //
  return estimate_covariances(points.data(), points.size(), k_neighbors);
}

}  // namespace gtsam_ext
