// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <gtsam_points/util/runnning_statistics.hpp>
#include <gtsam_points/ann/incremental_covariance_voxelmap.hpp>
#include <gtsam_points/ann/impl/incremental_voxelmap_impl.hpp>

namespace gtsam_points {

template class IncrementalVoxelMap<IncrementalCovarianceContainer>;

IncrementalCovarianceVoxelMap::IncrementalCovarianceVoxelMap(double voxel_resolution)
: IncrementalVoxelMap<IncrementalCovarianceContainer>(voxel_resolution) {
  num_neighbors = 10;
  min_num_neighbors = 5;
  warmup_cycles = 3;
  lowrate_cycles = 32;
  eig_stddev_thresh_scale = 1.0;
  num_threads = 1;

  eig_stats.resize(10);
}

IncrementalCovarianceVoxelMap::~IncrementalCovarianceVoxelMap() {}

void IncrementalCovarianceVoxelMap::insert(const PointCloud& points) {
  IncrementalVoxelMap<IncrementalCovarianceContainer>::insert(points);

  // Set flags for newborn points
  for (auto& voxel : flat_voxels) {
    for (auto& flags : voxel->second.flags) {
      if (flags == 0) {
        flags = lru_counter;
      }
    }
  }

  // Statistics of eigenvalues of past frames
  RunningStatistics<Eigen::Array3d> sum_stats;
  for (const auto& stats : eig_stats) {
    sum_stats += stats;
  }

  double eig1_min = 0.0;
  double eig2_max = std::numeric_limits<double>::max();

  if (sum_stats.size() > 1024) {
    const Eigen::Array3d mean = sum_stats.mean();
    const Eigen::Array3d stddev = sum_stats.var().sqrt();
    eig1_min = mean[0] - eig_stddev_thresh_scale * stddev[0];
    eig2_max = mean[1] + eig_stddev_thresh_scale * stddev[1];
  }

  std::vector<RunningStatistics<Eigen::Array3d>> new_stats(num_threads);
#pragma omp parallel for num_threads(num_threads) schedule(guided, 2)
  for (int k = 0; k < flat_voxels.size(); k++) {
    auto& voxel = flat_voxels[k];
    for (int i = 0; i < voxel->second.size(); i++) {
      // Check if the point needs to be updated.
      const bool in_warmup = (voxel->second.birthday(i) + warmup_cycles > lru_counter);
      const bool in_reeval = !voxel->second.valid(i) && ((lru_counter - voxel->second.birthday(i)) % lowrate_cycles) == 0;
      const bool do_update = in_warmup || in_reeval;

      if (!do_update) {
        continue;
      }

      // Find neighbors in the voxelmap.
      std::vector<size_t> k_indices(num_neighbors);
      std::vector<double> k_sq_dists(num_neighbors);
      size_t num_found = knn_search_force(voxel->second.points[i].data(), num_neighbors, k_indices.data(), k_sq_dists.data());
      if (num_found < min_num_neighbors) {
        continue;
      }

      // Calculate the covariance matrix.
      Eigen::Vector4d sum_points = Eigen::Vector4d::Zero();
      Eigen::Matrix4d sum_cross = Eigen::Matrix4d::Zero();
      for (int i = 0; i < num_found; i++) {
        const Eigen::Vector4d pt = this->point(k_indices[i]);
        sum_points += pt;
        sum_cross += pt * pt.transpose();
      }

      Eigen::Vector4d mean = sum_points / num_found;
      Eigen::Matrix4d cov = sum_cross - mean * sum_points.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig;
      eig.computeDirect(cov.block<3, 3>(0, 0));

      if (in_warmup) {
        new_stats[omp_get_thread_num()].add(eig.eigenvalues());
      }

      // Check if the normal is valid.
      if (eig.eigenvalues()[1] < eig1_min || eig.eigenvalues()[2] > eig2_max) {
        continue;
      }

      // Update the normal and the cov.
      voxel->second.normals[i] << eig.eigenvectors().col(0), 0.0;
      voxel->second.covs[i].block<3, 3>(0, 0) = eig.eigenvectors() * Eigen::Vector3d(1e-3, 1.0, 1.0).asDiagonal() * eig.eigenvectors().transpose();
      voxel->second.set_valid(i);
    }
  }

  // Update running statistics.
  RunningStatistics<Eigen::Array3d> sum_stats_new;
  for (const auto& stats : new_stats) {
    sum_stats_new += stats;
  }

  if (sum_stats_new.size() > 128) {
    eig_stats.pop_front();
    eig_stats.push_back(sum_stats_new);
  }
}

size_t IncrementalCovarianceVoxelMap::knn_search(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists) const {
  return IncrementalVoxelMap<IncrementalCovarianceContainer>::knn_search(pt, k, k_indices, k_sq_dists);
}

size_t IncrementalCovarianceVoxelMap::knn_search_force(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists) const {
  const Eigen::Vector4d query = (Eigen::Vector4d() << pt[0], pt[1], pt[2], 1.0).finished();
  const Eigen::Vector3i center = fast_floor(query * inv_leaf_size).template head<3>();

  size_t voxel_index = 0;
  const auto index_transform = [&](const size_t point_index) { return calc_index(voxel_index, point_index); };

  KnnResult<-1, decltype(index_transform)> result(k_indices, k_sq_dists, k, index_transform);
  for (const auto& offset : offsets) {
    const Eigen::Vector3i coord = center + offset;
    const auto found = voxels.find(coord);
    if (found == voxels.end()) {
      continue;
    }

    voxel_index = found->second;
    const auto& voxel = flat_voxels[voxel_index]->second;
    voxel.knn_search_force(query, result);
  }

  return result.num_found();
}

std::vector<Eigen::Vector4d> IncrementalCovarianceVoxelMap::voxel_points() const {
  std::vector<Eigen::Vector4d> points;
  points.reserve(num_voxels() * 10);

  for (const auto& voxel : flat_voxels) {
    for (int i = 0; i < voxel->second.size(); i++) {
      if (voxel->second.valid(i)) {
        points.emplace_back(voxel->second.points[i]);
      }
    }
  }

  return points;
}

std::vector<Eigen::Vector4d> IncrementalCovarianceVoxelMap::voxel_normals() const {
  std::vector<Eigen::Vector4d> normals;
  normals.reserve(num_voxels() * 10);

  for (const auto& voxel : flat_voxels) {
    for (int i = 0; i < voxel->second.size(); i++) {
      if (voxel->second.valid(i)) {
        normals.emplace_back(voxel->second.normals[i]);
      }
    }
  }

  return normals;
}

std::vector<Eigen::Matrix4d> IncrementalCovarianceVoxelMap::voxel_covs() const {
  std::vector<Eigen::Matrix4d> covs;
  covs.reserve(num_voxels() * 10);

  for (const auto& voxel : flat_voxels) {
    for (int i = 0; i < voxel->second.size(); i++) {
      if (voxel->second.valid(i)) {
        covs.emplace_back(voxel->second.covs[i]);
      }
    }
  }

  return covs;
}

}  // namespace gtsam_points