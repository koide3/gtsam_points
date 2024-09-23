// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/intensity_gradients_ivox.hpp>

#include <Eigen/Eigen>
#include <gtsam_points/config.hpp>
#include <gtsam_points/util/parallelism.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/parallel_for.h>
#endif

namespace gtsam_points {

IntensityGradientsiVox::IntensityGradientsiVox(const double voxel_resolution, const double min_points_dist, int k_neighbors, int num_threads)
: iVox(voxel_resolution, min_points_dist),
  k_neighbors(k_neighbors),
  num_threads(num_threads) {}

IntensityGradientsiVox::~IntensityGradientsiVox() {}

void IntensityGradientsiVox::insert(const PointCloud& frame) {
  iVox::insert(frame);

  // Remove gradient voxels that are already removed from iVox
  for (auto grad = gradmap.begin(); grad != gradmap.end();) {
    if (voxelmap.find(grad->first) == voxelmap.end()) {
      grad = gradmap.erase(grad);
    } else {
      grad++;
    }
  }

  // Find gradient voxels corresponding to iVox voxels
  std::vector<std::pair<Eigen::Vector3i, LinearContainer::Ptr>> flat_voxels(voxelmap.begin(), voxelmap.end());
  std::vector<LinearContainer::Ptr> flat_grads(flat_voxels.size());
  grads.resize(voxels.size());

  for (int i = 0; i < flat_voxels.size(); i++) {
    const auto& voxel = flat_voxels[i];
    const auto& coord = voxel.first;
    auto found = gradmap.find(coord);

    if (found == gradmap.end()) {
      found = gradmap.insert(found, std::make_pair(coord, std::make_shared<LinearContainer>(0)));
    }

    flat_grads[i] = found->second;
    grads[voxel.second->serial_id] = found->second;
  }

  const auto pervoxel_task = [&](int i) {
    const auto& voxel = flat_voxels[i];
    const auto& gradients = flat_grads[i];
    gradients->points.reserve(voxel.second->size());
    gradients->normals.reserve(voxel.second->size());

    std::vector<size_t> k_indices(k_neighbors);
    std::vector<double> k_sq_dists(k_neighbors);

    for (int j = gradients->size(); j < voxel.second->size(); j++) {
      const auto& point = voxel.second->points[j];
      const double intensity = voxel.second->intensities[j];

      const int num_found = knn_search(point.data(), k_neighbors, k_indices.data(), k_sq_dists.data());

      const Eigen::Vector4d normal = estimate_normal(point, num_found, k_indices.data(), k_sq_dists.data());
      const Eigen::Vector4d gradient = estimate_gradient(point, normal, intensity, num_found, k_indices.data(), k_sq_dists.data());
      gradients->normals.push_back(normal);
      gradients->points.push_back(gradient);
    }
  };

  if (is_omp_default() || num_threads == 1) {
    // Add new points and estimate their normal and gradients
#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
    for (int i = 0; i < flat_voxels.size(); i++) {
      pervoxel_task(i);
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, flat_voxels.size(), 8), [&](const tbb::blocked_range<int>& range) {
      for (int i = range.begin(); i < range.end(); i++) {
        pervoxel_task(i);
      }
    });
#else
    std::cerr << "error: TBB is not available" << std::endl;
    abort();
#endif
  }
}

Eigen::Vector4d
IntensityGradientsiVox::estimate_normal(const Eigen::Vector4d& point, const int num_found, const size_t* k_indices, const double* k_sq_dists) const {
  if (num_found < 3) {
    return (Eigen::Vector4d() << Eigen::Vector3d::Random().normalized(), 0.0).finished();
  }

  Eigen::Vector4d sum_points = Eigen::Vector4d::Zero();
  Eigen::Matrix4d sum_covs = Eigen::Matrix4d::Zero();

  for (int i = 0; i < num_found; i++) {
    const auto& pt = frame::point<iVox>(*this, k_indices[i]);
    sum_points += pt;
    sum_covs += pt * pt.transpose();
  }

  const Eigen::Vector4d mean = sum_points / num_found;
  const Eigen::Matrix4d cov = (sum_covs - mean * sum_points.transpose()) / num_found;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig;
  eig.computeDirect(cov.block<3, 3>(0, 0));

  Eigen::Vector4d normal = (Eigen::Vector4d() << eig.eigenvectors().col(0), 0.0).finished();
  if (point.dot(normal) > 1.0) {
    normal = -normal;
  }

  return normal;
}

Eigen::Vector4d IntensityGradientsiVox::estimate_gradient(
  const Eigen::Vector4d& point,
  const Eigen::Vector4d& normal,
  const double intensity,
  const int num_found,
  const size_t* k_indices,
  const double* k_sq_dists) const {
  //
  if (num_found < 3) {
    return Eigen::Vector4d::Zero();
  }

  Eigen::Matrix<double, -1, 4> A = Eigen::Matrix<double, -1, 4>::Zero(num_found, 4);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(num_found);

  A.row(0) = normal;
  b[0] = 0.0;

  for (int i = 1; i < num_found; i++) {
    const auto& point_ = frame::point<iVox>(*this, k_indices[i]);
    const double intensity_ = frame::intensity<iVox>(*this, k_indices[i]);
    const Eigen::Vector4d projected = point_ - (point_ - point).dot(normal) * normal;
    A.row(i) = projected - point;
    b(i) = (intensity_ - intensity);
  }

  Eigen::Matrix3d H = (A.transpose() * A).block<3, 3>(0, 0);
  Eigen::Vector3d e = (A.transpose() * b).head<3>();

  return (Eigen::Vector4d() << H.inverse() * e, 0.0).finished();
}

const Eigen::Vector4d& IntensityGradientsiVox::normal(const size_t i) const {
  return grads[voxel_id(i)]->normals[point_id(i)];
}

const Eigen::Vector4d& IntensityGradientsiVox::intensity_gradient(const size_t i) const {
  return grads[voxel_id(i)]->points[point_id(i)];
}

std::vector<Eigen::Vector4d> IntensityGradientsiVox::voxel_normals() const {
  std::vector<Eigen::Vector4d> normals;
  for (const auto& grad : grads) {
    normals.insert(normals.end(), grad->normals.begin(), grad->normals.end());
  }

  return normals;
}

std::vector<double> IntensityGradientsiVox::voxel_intensities() const {
  std::vector<double> intensities;
  for (const auto& voxel : voxels) {
    intensities.insert(intensities.end(), voxel->intensities.begin(), voxel->intensities.end());
  }

  return intensities;
}

std::vector<Eigen::Vector4d> IntensityGradientsiVox::voxel_intensity_gradients() const {
  std::vector<Eigen::Vector4d> gradients;
  for (const auto& grad : grads) {
    gradients.insert(gradients.end(), grad->points.begin(), grad->points.end());
  }
  return gradients;
}

}  // namespace gtsam_points