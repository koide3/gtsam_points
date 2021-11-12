// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_ext/factors/intensity_gradients.hpp>

#include <Eigen/Eigen>

#include <iostream>
#include <gtsam_ext/ann/kdtree.hpp>

namespace gtsam_ext {

IntensityGradients::Ptr IntensityGradients::estimate(const gtsam_ext::Frame::ConstPtr& frame, int k_neighbors, int num_threads) {
  if (!frame->has_points() || !frame->has_normals() || !frame->has_intensities()) {
    std::cerr << "error: input frame doesn't have required attributes for intensity gradient estimation!!" << std::endl;
    abort();
  }

  gtsam_ext::KdTree kdtree(frame->points, frame->size());

  IntensityGradients::Ptr gradients(new IntensityGradients);
  gradients->intensity_gradients.resize(frame->size());

#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
  for (int i = 0; i < frame->size(); i++) {
    std::vector<size_t> k_indices(k_neighbors);
    std::vector<double> k_sq_dists(k_neighbors);
    kdtree.knn_search(frame->points[i].data(), k_neighbors, k_indices.data(), k_sq_dists.data());

    // Estimate color gradient
    const auto& point = frame->points[i];
    const auto& normal = frame->normals[i];
    const double intensity = frame->intensities[i];

    Eigen::Matrix<double, -1, 4> A = Eigen::Matrix<double, -1, 4>::Zero(k_neighbors, 4);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(k_neighbors);

    // dp^T np = 0
    A.row(0) = normal;
    b[0] = 0.0;

    // Intensity gradient in the tangent space
    for (int j = 1; j < k_neighbors; j++) {
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
  }

  return gradients;
}

IntensityGradients::Ptr IntensityGradients::estimate(const gtsam_ext::FrameCPU::Ptr& frame, int k_neighbors, int num_threads) {
  return estimate(frame, k_neighbors, k_neighbors, num_threads);
}

IntensityGradients::Ptr IntensityGradients::estimate(const gtsam_ext::FrameCPU::Ptr& frame, int k_geom_neighbors, int k_photo_neighbors, int num_threads) {
  gtsam_ext::KdTree kdtree(frame->points, frame->size());

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

#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
  for (int i = 0; i < frame->size(); i++) {
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
  }

  return gradients;
}

}  // namespace gtsam_ext