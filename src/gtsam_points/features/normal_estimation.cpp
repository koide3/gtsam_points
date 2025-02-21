// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/features/normal_estimation.hpp>

#include <iostream>
#include <Eigen/Eigen>
#include <gtsam_points/config.hpp>
#include <gtsam_points/util/parallelism.hpp>
#include <gtsam_points/features/covariance_estimation.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/parallel_for.h>
#endif

namespace gtsam_points {

std::vector<Eigen::Vector4d> estimate_normals(const Eigen::Vector4d* points, const Eigen::Matrix4d* covs, int num_points, int num_threads) {
  std::vector<Eigen::Vector4d> normals(num_points, Eigen::Vector4d::Zero());

  const auto perpoint_task = [&](int i) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig;
    eig.computeDirect(covs[i].block<3, 3>(0, 0));
    normals[i].head<3>() = eig.eigenvectors().col(0);

    if (points[i].dot(normals[i]) > 1.0) {
      normals[i] = -normals[i];
    }
  };

  if (is_omp_default() || num_threads == 1) {
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_points; i++) {
      perpoint_task(i);
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, num_points, 64), [&](const tbb::blocked_range<int>& range) {
      for (int i = range.begin(); i < range.end(); i++) {
        perpoint_task(i);
      }
    });
#else
    std::cerr << "error : TBB is not enabled" << std::endl;
    abort();
#endif
  }

  return normals;
}

std::vector<Eigen::Vector4d> estimate_normals(const Eigen::Vector4d* points, int num_points, int k_neighbors, int num_threads) {
  auto covs = estimate_covariances(points, num_points, k_neighbors, num_threads);
  return estimate_normals(points, covs.data(), num_points);
}

std::vector<Eigen::Vector4d> estimate_normals(const PointCloud& points, int k_neighbors, int num_threads) {
  if (points.has_covs()) {
    return gtsam_points::estimate_normals(points.points, points.covs, points.size(), num_threads);
  }
  return gtsam_points::estimate_normals(points.points, points.size(), k_neighbors, num_threads);
}

}  // namespace gtsam_points