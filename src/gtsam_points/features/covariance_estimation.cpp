// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/features/covariance_estimation.hpp>

#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <gtsam_points/config.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/util/parallelism.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/parallel_for.h>
#endif

namespace gtsam_points {

std::vector<Eigen::Matrix4d> estimate_covariances(const Eigen::Vector4d* points, int num_points, const CovarianceEstimationParams& params) {
  KdTree tree(points, num_points, params.num_threads);
  std::vector<Eigen::Matrix4d> covs(num_points);

  const auto perpoint_task = [&](int i) {
    std::vector<size_t> k_indices(params.k_neighbors);
    std::vector<double> k_sq_dists(params.k_neighbors);
    size_t num_found = tree.knn_search(points[i].data(), params.k_neighbors, &k_indices[0], &k_sq_dists[0]);

    if (num_found < params.k_neighbors) {
      std::cerr << "warning: fewer than k neighbors found for point " << i << std::endl;
      covs[i].setIdentity();
      return;
    }

    Eigen::Vector4d sum_points = Eigen::Vector4d::Zero();
    Eigen::Matrix4d sum_covs = Eigen::Matrix4d::Zero();

    for (int j = 0; j < num_found; j++) {
      const auto& pt = points[k_indices[j]];
      sum_points += pt;
      sum_covs += pt * pt.transpose();
    }

    Eigen::Vector4d mean = sum_points / num_found;
    Eigen::Matrix4d cov = (sum_covs - mean * sum_points.transpose()) / num_found;

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
  };

  if (is_omp_default() || params.num_threads == 1) {
#pragma omp parallel for num_threads(params.num_threads) schedule(guided, 8)
    for (int i = 0; i < num_points; i++) {
      perpoint_task(i);
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, num_points, 8), [&](const tbb::blocked_range<int>& range) {
      for (int i = range.begin(); i < range.end(); i++) {
        perpoint_task(i);
      }
    });
#else
    std::cerr << "error: TBB is not available" << std::endl;
    abort();
#endif
  }

  return covs;
}

std::vector<Eigen::Matrix4d> estimate_covariances(const Eigen::Vector4d* points, int num_points, int k_neighbors, int num_threads) {
  CovarianceEstimationParams params;
  params.k_neighbors = k_neighbors;
  params.num_threads = num_threads;
  return estimate_covariances(points, num_points, params);
}

std::vector<Eigen::Matrix4d>
estimate_covariances(const Eigen::Vector4d* points, int num_points, int k_neighbors, const Eigen::Vector3d& eigen_values, int num_threads) {
  CovarianceEstimationParams params;
  params.k_neighbors = k_neighbors;
  params.eigen_values = eigen_values;
  params.num_threads = num_threads;
  return estimate_covariances(points, num_points, params);
}

std::vector<Eigen::Matrix4d> estimate_covariances(const PointCloud& points, int k_neighbors, int num_threads) {
  return estimate_covariances(points.points, points.num_points, k_neighbors, num_threads);
}

}  // namespace gtsam_points
