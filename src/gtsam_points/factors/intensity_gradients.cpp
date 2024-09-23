// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/intensity_gradients.hpp>

#include <Eigen/Eigen>

#include <iostream>
#include <gtsam_points/config.hpp>
#include <gtsam_points/ann/kdtree2.hpp>
#include <gtsam_points/util/parallelism.hpp>
#include <gtsam_points/types/frame_traits.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/parallel_for.h>
#endif

namespace gtsam_points {

IntensityGradients::Ptr IntensityGradients::estimate(const PointCloud::ConstPtr& frame, const std::vector<int>& neighbors, int k_photo_neighbors) {
  if (!frame::has_points(*frame) || !frame::has_normals(*frame) || !frame::has_intensities(*frame)) {
    std::cerr << "error: input frame doesn't have required attributes for intensity gradient estimation!!" << std::endl;
    abort();
  }

  const int k = neighbors.size() / frame::size(*frame);
  if (frame::size(*frame) * k != neighbors.size()) {
    std::cerr << "error: k * frame->size() != neighbors.size()!!" << std::endl;
    abort();
  }

  if (k_photo_neighbors > k) {
    std::cerr << "error: k_photo_neighbors > k!!" << std::endl;
    abort();
  }

  IntensityGradients::Ptr gradients(new IntensityGradients);
  gradients->intensity_gradients.resize(frame::size(*frame));

  for (int i = 0; i < frame::size(*frame); i++) {
    // Estimate color gradient
    const auto& point = frame::point(*frame, i);
    const auto& normal = frame::normal(*frame, i);
    const double intensity = frame::intensity(*frame, i);

    Eigen::Matrix<double, -1, 4> A = Eigen::Matrix<double, -1, 4>::Zero(k_photo_neighbors, 4);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(k_photo_neighbors);

    // dp^T np = 0
    A.row(0) = normal;
    b[0] = 0.0;

    // Intensity gradient in the tangent space
    for (int j = 1; j < k_photo_neighbors; j++) {
      const int index = neighbors[k * i + j];
      const auto& point_ = frame::point(*frame, index);
      const double intensity_ = frame::intensity(*frame, index);
      const Eigen::Vector4d projected = point_ - (point_ - point).dot(normal) * normal;
      A.row(j) = projected - point;
      b(j) = (intensity_ - intensity);
    }

    Eigen::Matrix3d H = (A.transpose() * A).block<3, 3>(0, 0);
    Eigen::Vector3d e = (A.transpose() * b).head<3>();
    gradients->intensity_gradients[i] << H.inverse() * e, 0.0;
  }

  return gradients;
}

IntensityGradients::Ptr IntensityGradients::estimate(const PointCloud::ConstPtr& frame, int k_neighbors, int num_threads) {
  if (!frame::has_points(*frame) || !frame::has_normals(*frame) || !frame::has_intensities(*frame)) {
    std::cerr << "error: input frame doesn't have required attributes for intensity gradient estimation!!" << std::endl;
    abort();
  }

  gtsam_points::KdTree2<PointCloud> kdtree(frame);

  IntensityGradients::Ptr gradients(new IntensityGradients);
  gradients->intensity_gradients.resize(frame::size(*frame));

  const auto perpoint_task = [&](int i) {
    std::vector<size_t> k_indices(k_neighbors);
    std::vector<double> k_sq_dists(k_neighbors);
    kdtree.knn_search(frame::point(*frame, i).data(), k_neighbors, k_indices.data(), k_sq_dists.data());

    // Estimate color gradient
    const auto& point = frame::point(*frame, i);
    const auto& normal = frame::normal(*frame, i);
    const double intensity = frame::intensity(*frame, i);

    Eigen::Matrix<double, -1, 4> A = Eigen::Matrix<double, -1, 4>::Zero(k_neighbors, 4);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(k_neighbors);

    // dp^T np = 0
    A.row(0) = normal;
    b[0] = 0.0;

    // Intensity gradient in the tangent space
    for (int j = 1; j < k_neighbors; j++) {
      const int index = k_indices[j];
      const auto& point_ = frame::point(*frame, index);
      const double intensity_ = frame::intensity(*frame, index);
      const Eigen::Vector4d projected = point_ - (point_ - point).dot(normal) * normal;
      A.row(j) = projected - point;
      b(j) = (intensity_ - intensity);
    }

    Eigen::Matrix3d H = (A.transpose() * A).block<3, 3>(0, 0);
    Eigen::Vector3d e = (A.transpose() * b).head<3>();
    gradients->intensity_gradients[i] << H.inverse() * e, 0.0;
  };

  if (is_omp_default() || num_threads == 1) {
#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
    for (int i = 0; i < frame->size(); i++) {
      perpoint_task(i);
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, frame->size(), 8), [&](const tbb::blocked_range<int>& range) {
      for (int i = range.begin(); i < range.end(); i++) {
        perpoint_task(i);
      }
    });
#else
    std::cerr << "error: TBB is not available" << std::endl;
    abort();
#endif
  }

  return gradients;
}

IntensityGradients::Ptr
IntensityGradients::estimate(const gtsam_points::PointCloudCPU::Ptr& frame, int k_geom_neighbors, int k_photo_neighbors, int num_threads) {
  gtsam_points::KdTree2<PointCloud> kdtree(frame, num_threads);

  bool estimate_normals = frame->normals == nullptr;
  bool estimate_covs = frame->covs == nullptr;

  if (estimate_normals) {
    frame->normals_storage.resize(frame->size());
    frame->normals = frame->normals_storage.data();
  }

  if (estimate_covs) {
    frame->covs_storage.resize(frame->size());
    frame->covs = frame->covs_storage.data();
  }

  IntensityGradients::Ptr gradients(new IntensityGradients);
  gradients->intensity_gradients.resize(frame->size());

  const int k_neighbors = std::max(k_geom_neighbors, k_photo_neighbors);

  const auto perpoint_task = [&](int i) {
    std::vector<size_t> k_indices(k_neighbors);
    std::vector<double> k_sq_dists(k_neighbors);
    kdtree.knn_search(frame->points[i].data(), k_neighbors, k_indices.data(), k_sq_dists.data());

    // Estimate normals and covariances
    if (estimate_normals || estimate_covs) {
      Eigen::Vector4d sum_pts = Eigen::Vector4d::Zero();
      Eigen::Matrix4d sum_cross = Eigen::Matrix4d::Zero();

      for (int j = 0; j < k_geom_neighbors; j++) {
        const int index = k_indices[j];
        const auto& pt = frame->points[index];
        sum_pts += pt;
        sum_cross += pt * pt.transpose();
      }

      Eigen::Vector4d mean = sum_pts / k_geom_neighbors;
      Eigen::Matrix4d cov = (sum_cross - mean * sum_pts.transpose()) / k_geom_neighbors;

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig;
      eig.computeDirect(cov.block<3, 3>(0, 0));

      // Normal estimation
      if (estimate_normals) {
        frame->normals[i] << eig.eigenvectors().col(0).normalized(), 0.0;
        if (frame->normals[i].dot(frame->points[i]) > 0.0) {
          frame->normals[i] = -frame->normals[i];
        }
      }

      // Covariance estimation
      if (estimate_covs) {
        Eigen::Vector3d values(1e-3, 1.0, 1.0);
        frame->covs[i].setZero();
        frame->covs[i].block<3, 3>(0, 0) = eig.eigenvectors() * values.asDiagonal() * eig.eigenvectors().inverse();
      }
    }

    // Estimate color gradient
    const auto& point = frame->points[i];
    const auto& normal = frame->normals[i];
    const double intensity = frame->intensities[i];

    Eigen::Matrix<double, -1, 4> A = Eigen::Matrix<double, -1, 4>::Zero(k_photo_neighbors, 4);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(k_photo_neighbors);

    // dp^T np = 0
    A.row(0) = normal;
    b[0] = 0.0;

    // Intensity gradient in the tangent space
    for (int j = 1; j < k_photo_neighbors; j++) {
      const int index = k_indices[j];
      const auto& point_ = frame->points[index];
      const double intensity_ = frame->intensities[index];
      const Eigen::Vector4d projected = point_ - (point_ - point).dot(normal) * normal;
      A.row(j) = projected - point;
      b(j) = (intensity_ - intensity);
    }

    Eigen::Matrix3d H = (A.transpose() * A).block<3, 3>(0, 0);
    Eigen::Vector3d e = (A.transpose() * b).head<3>();
    gradients->intensity_gradients[i] << H.inverse() * e, 0.0;
  };

  if (is_omp_default() || num_threads == 1) {
#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
    for (int i = 0; i < frame->size(); i++) {
      perpoint_task(i);
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, frame->size(), 8), [&](const tbb::blocked_range<int>& range) {
      for (int i = range.begin(); i < range.end(); i++) {
        perpoint_task(i);
      }
    });
#else
    std::cerr << "error: TBB is not available" << std::endl;
    abort();
#endif
  }

  return gradients;
}

}  // namespace gtsam_points