// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#include <numeric>
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <gtsam_points/util/runnning_statistics.hpp>
#include <gtsam_points/ann/incremental_covariance_voxelmap.hpp>
#include <gtsam_points/ann/impl/incremental_voxelmap_impl.hpp>
#include <gtsam_points/util/easy_profiler.hpp>

namespace gtsam_points {

template class IncrementalVoxelMap<IncrementalCovarianceContainer>;

IncrementalCovarianceVoxelMap::IncrementalCovarianceVoxelMap(double voxel_resolution)
: IncrementalVoxelMap<IncrementalCovarianceContainer>(voxel_resolution) {
  num_neighbors = 20;
  min_num_neighbors = 5;
  warmup_cycles = 3;
  lowrate_cycles = 16;  // Must be power of 2
  remove_invalid_age_thresh = 64;
  eig_stddev_thresh_scale = 1.0;
  num_threads = 1;

  eig_stats.resize(10);
}

IncrementalCovarianceVoxelMap::~IncrementalCovarianceVoxelMap() {}

void IncrementalCovarianceVoxelMap::set_num_neighbors(int num_neighbors) {
  this->num_neighbors = num_neighbors;
}

void IncrementalCovarianceVoxelMap::set_min_num_neighbors(int min_num_neighbors) {
  this->min_num_neighbors = min_num_neighbors;
}

void IncrementalCovarianceVoxelMap::set_warmup_cycles(int warmup_cycles) {
  this->warmup_cycles = warmup_cycles;
}

void IncrementalCovarianceVoxelMap::set_lowrate_cycles(int lowrate_cycles) {
  if (lowrate_cycles & (lowrate_cycles - 1)) {
    std::cerr << "warning: lowrate_cycles must be a power of 2." << std::endl;
    return;
  }

  this->lowrate_cycles = lowrate_cycles;
}

void IncrementalCovarianceVoxelMap::set_remove_invalid_age_thresh(int remove_invalid_age_thresh) {
  this->remove_invalid_age_thresh = remove_invalid_age_thresh;
}

void IncrementalCovarianceVoxelMap::set_eig_stddev_thresh_scale(double eig_stddev_thresh_scale) {
  this->eig_stddev_thresh_scale = eig_stddev_thresh_scale;
}

void IncrementalCovarianceVoxelMap::set_num_threads(int num_threads) {
  this->num_threads = num_threads;
}

void IncrementalCovarianceVoxelMap::clear() {
  IncrementalVoxelMap<IncrementalCovarianceContainer>::clear();
  eig_stats.clear();
  eig_stats.resize(10);
}

void IncrementalCovarianceVoxelMap::insert(const PointCloud& points) {
  EasyProfiler prof("IncrementalCovarianceVoxelMap::insert");

  prof.push("insert");
  IncrementalVoxelMap<IncrementalCovarianceContainer>::insert(points);

  prof.push("prepare");

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

  prof.push("estimate normals");

  std::vector<RunningStatistics<Eigen::Array3d>> new_stats(num_threads);
#pragma omp parallel for num_threads(num_threads) schedule(guided, 2)
  for (int k = 0; k < flat_voxels.size(); k++) {
    auto& voxel = flat_voxels[k];

    for (int i = 0; i < voxel->second.size(); i++) {
      // Check if the point needs to be updated.
      const bool in_warmup = (voxel->second.birthday(i) + warmup_cycles > lru_counter);
      const bool in_reeval = !voxel->second.valid(i) && ((lru_counter - voxel->second.birthday(i) + 1) & (lowrate_cycles - 1)) == 0;
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

      int thread_num = 0;
#ifdef _OPENMP
      thread_num = omp_get_thread_num();
#endif

      if (in_warmup) {
        new_stats[thread_num].add(eig.eigenvalues());
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

  if (sum_stats_new.size() > 32) {
    eig_stats.pop_front();
    eig_stats.push_back(sum_stats_new);
  }

  prof.push("remove old invalid");

  if (lru_counter % lru_clear_cycle == 0) {
#pragma omp parallel for num_threads(num_threads) schedule(guided, 2)
    for (size_t i = 0; i < flat_voxels.size(); i++) {
      auto& voxel = flat_voxels[i];
      voxel->second.remove_old_invalid(remove_invalid_age_thresh, lru_counter);
    }

    auto remove_loc = std::remove_if(flat_voxels.begin(), flat_voxels.end(), [](const auto& voxel) { return voxel->second.size() == 0; });
    const bool needs_rehash = remove_loc != flat_voxels.end();
    flat_voxels.erase(remove_loc, flat_voxels.end());

    if (needs_rehash) {
      voxels.clear();
      for (const auto& voxel : flat_voxels) {
        voxels[voxel->first.coord] = voxels.size();
      }
    }
  }

  prof.push("done");
}

size_t IncrementalCovarianceVoxelMap::knn_search(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists, double max_sq_dist) const {
  return IncrementalVoxelMap<IncrementalCovarianceContainer>::knn_search(pt, k, k_indices, k_sq_dists, max_sq_dist);
}

size_t IncrementalCovarianceVoxelMap::knn_search_force(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists, double max_sq_dist) const {
  const Eigen::Vector4d query = (Eigen::Vector4d() << pt[0], pt[1], pt[2], 1.0).finished();
  const Eigen::Vector3i center = fast_floor(query * inv_leaf_size).template head<3>();

  size_t voxel_index = 0;
  const auto index_transform = [&](const size_t point_index) { return calc_index(voxel_index, point_index); };

  KnnResult<-1, decltype(index_transform)> result(k_indices, k_sq_dists, k, index_transform, max_sq_dist);
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

std::vector<size_t> IncrementalCovarianceVoxelMap::valid_indices(int num_threads) const {
  if (num_threads < 0) {
    num_threads = this->num_threads;
  }

  std::vector<std::vector<size_t>> valid_indices(num_threads);
  for (auto& indices : valid_indices) {
    indices.reserve(num_voxels() * 10 / num_threads);
  }

#pragma omp parallel for num_threads(num_threads)
  for (size_t i = 0; i < flat_voxels.size(); i++) {
    const auto& voxel = flat_voxels[i];
    for (int j = 0; j < voxel->second.size(); j++) {
      if (voxel->second.valid(j)) {
        int thread_num = 0;
#ifdef _OPENMP
        thread_num = omp_get_thread_num();
#endif
        valid_indices[thread_num].push_back(calc_index(i, j));
      }
    }
  }

  const size_t num_points =
    std::accumulate(valid_indices.begin(), valid_indices.end(), 0, [](const size_t sum, const auto& indices) { return sum + indices.size(); });
  valid_indices[0].reserve(num_points);

  for (int i = 1; i < valid_indices.size(); i++) {
    valid_indices[0].insert(valid_indices[0].end(), valid_indices[i].begin(), valid_indices[i].end());
  }

  return std::move(valid_indices[0]);
}

std::vector<Eigen::Vector4d> IncrementalCovarianceVoxelMap::voxel_points(const std::vector<size_t>& indices) const {
  std::vector<Eigen::Vector4d> points(indices.size());
  std::transform(indices.begin(), indices.end(), points.begin(), [&](const size_t i) {
    return flat_voxels[voxel_id(i)]->second.points[point_id(i)];
  });
  return points;
}

std::vector<Eigen::Vector4d> IncrementalCovarianceVoxelMap::voxel_normals(const std::vector<size_t>& indices) const {
  std::vector<Eigen::Vector4d> normals(indices.size());
  std::transform(indices.begin(), indices.end(), normals.begin(), [&](const size_t i) {
    return flat_voxels[voxel_id(i)]->second.normals[point_id(i)];
  });
  return normals;
}

std::vector<Eigen::Matrix4d> IncrementalCovarianceVoxelMap::voxel_covs(const std::vector<size_t>& indices) const {
  std::vector<Eigen::Matrix4d> covs(indices.size());
  std::transform(indices.begin(), indices.end(), covs.begin(), [&](const size_t i) { return flat_voxels[voxel_id(i)]->second.covs[point_id(i)]; });
  return covs;
}

std::vector<Eigen::Vector4d> IncrementalCovarianceVoxelMap::voxel_points() const {
  return voxel_points(valid_indices());
}

std::vector<Eigen::Vector4d> IncrementalCovarianceVoxelMap::voxel_normals() const {
  return voxel_normals(valid_indices());
}

std::vector<Eigen::Matrix4d> IncrementalCovarianceVoxelMap::voxel_covs() const {
  return voxel_covs(valid_indices());
}

}  // namespace gtsam_points