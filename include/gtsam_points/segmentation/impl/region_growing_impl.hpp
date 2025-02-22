// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <mutex>
#include <numeric>
#include <unordered_set>
#include <gtsam_points/util/easy_profiler.hpp>
#include <gtsam_points/segmentation/region_growing.hpp>

namespace gtsam_points {

template <typename PointCloud>
RegionGrowingContext region_growing_init_(
  const PointCloud& points,
  const NearestNeighborSearch& search,
  const Eigen::Vector4d& seed_point,
  const RegionGrowingParams& params) {
  //
  size_t index;
  double sq_distance;
  // Find the closest point to be the seed
  if (!search.knn_search(seed_point.data(), 1, &index, &sq_distance)) {
    return RegionGrowingContext();
  }

  RegionGrowingContext context;
  context.seed_points.emplace_back(index);
  context.visited_seeds.resize(frame::size(points), 0);
  return context;
}

template <typename PointCloud>
bool region_growing_step_(
  RegionGrowingContext& context,
  const PointCloud& points,
  const NearestNeighborSearch& search,
  const RegionGrowingParams& params) {
  //
  if (frame::has_normals(points) == false) {
    std::cerr << "warning: normals are required for region growing" << std::endl;
    return true;
  }

  if (context.visited_seeds.size() != frame::size(points)) {
    std::cerr << "error: context.visited_seeds.size() != frame::size(points)" << std::endl;
    return true;
  }

  if (context.seed_points.empty()) {
    return true;
  }

  if (context.cluster_indices.size() > params.max_cluster_size) {
    context.seed_points.clear();
    return true;
  }

  const int seed_index = context.seed_points.front();
  context.seed_points.pop_front();

  if (context.visited_seeds[seed_index]) {
    // Skip if the seed has been visited
    return region_growing_step_(context, points, search, params);
  }

  // Add the seed to the cluster
  context.visited_seeds[seed_index] = 1;
  context.cluster_indices.emplace_back(seed_index);

  // Find neighbors of the seed
  std::vector<size_t> neighbor_indices;
  std::vector<double> neighbor_sq_dists;
  search
    .radius_search(frame::point(points, seed_index).data(), params.distance_threshold, neighbor_indices, neighbor_sq_dists, params.max_cluster_size);

  const double sq_distance_threshold = params.distance_threshold * params.distance_threshold;
  const double cosine_threshold = std::cos(params.angle_threshold);
  for (size_t i = 0; i < neighbor_indices.size(); i++) {
    // Skip if the neighbor has been visited
    if (context.visited_seeds[neighbor_indices[i]]) {
      continue;
    }

    // Distance check
    if (neighbor_sq_dists[i] > sq_distance_threshold) {
      continue;
    }

    // Angle check
    const auto& seed_normal = frame::normal(points, seed_index);
    const auto& pt_normal = frame::normal(points, neighbor_indices[i]);
    if (seed_normal.dot(pt_normal) < cosine_threshold) {
      continue;
    }

    // Add the neighbor to the seed list
    context.seed_points.emplace_back(neighbor_indices[i]);
  }

  return false;
}

template <typename PointCloud>
void region_growing_dilation_(
  RegionGrowingContext& context,
  const PointCloud& points,
  const NearestNeighborSearch& search,
  const RegionGrowingParams& params) {
  //
  std::vector<std::unordered_set<size_t>> new_indices(params.num_threads);

  // Find points within the dilation radius of the cluster
#pragma omp parallel for num_threads(params.num_threads) schedule(guided, 4)
  for (size_t i = 0; i < context.cluster_indices.size(); i++) {
    std::vector<size_t> indices;
    std::vector<double> sq_dists;
    search.radius_search(frame::point(points, context.cluster_indices[i]).data(), params.dilation_radius, indices, sq_dists);

    new_indices[omp_get_thread_num()].insert(indices.begin(), indices.end());
  }

  // Merge and sort the indices
  for (size_t i = 1; i < new_indices.size(); i++) {
    new_indices[0].insert(new_indices[i].begin(), new_indices[i].end());
  }

  context.cluster_indices.assign(new_indices[0].begin(), new_indices[0].end());
  std::sort(context.cluster_indices.begin(), context.cluster_indices.end());
}

template <typename PointCloud>
bool region_growing_update_(
  RegionGrowingContext& context,
  const PointCloud& points,
  const NearestNeighborSearch& search,
  const RegionGrowingParams& params) {
  //
  if (context.seed_points.empty()) {
    return true;
  }

  for (int i = 0; i < params.max_steps; i++) {
    if (context.seed_points.empty()) {
      break;
    }

    region_growing_step_(context, points, search, params);
  }

  if (context.seed_points.empty() && params.dilation_radius > 0.0) {
    region_growing_dilation_(context, points, search, params);
  }

  return context.seed_points.empty();
}

}  // namespace gtsam_points
