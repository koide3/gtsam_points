// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <deque>
#include <vector>
#include <cstdint>
#include <Eigen/Core>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/ann/nearest_neighbor_search.hpp>

namespace gtsam_points {

/// @brief Region growing parameters
struct RegionGrowingParams {
  double distance_threshold = 0.5;               ///< Distance threshold
  double angle_threshold = 10.0 * M_PI / 180.0;  ///< Angle threshold
  double dilation_radius = 0.5;                  ///< Radius of dilation after region growing
  int max_cluster_size = 1000000;                ///< Maximum cluster size
  int max_steps = 1000000;                       ///< Maximum number of update steps
  int num_threads = 1;                           ///< Number of threads
};

/// @brief Region growing context
struct RegionGrowingContext {
  std::vector<size_t> cluster_indices;  ///< Indices of points in the cluster
  std::deque<size_t> seed_points;       ///< Seed points to be evaluated
  std::vector<bool> visited_seeds;      ///< Seed points that have been visited
};

/// @brief Initialize region growing
/// @param points       Point cloud
/// @param search       Nearest neighbor search
/// @param seed_point   Seed point
/// @param params       Region growing parameters
/// @return             Region growing context
template <typename PointCloud>
RegionGrowingContext region_growing_init_(
  const PointCloud& points,
  const NearestNeighborSearch& search,
  const Eigen::Vector4d& seed_point,
  const RegionGrowingParams& params);

/// @brief Update the region growing context once
/// @param context    Region growing context
/// @param points     Point cloud
/// @param search     Nearest neighbor search
/// @param params     Region growing parameters
/// @return           True if the region growing is converged
template <typename PointCloud>
bool region_growing_step_(
  RegionGrowingContext& context,
  const PointCloud& points,
  const NearestNeighborSearch& search,
  const RegionGrowingParams& params);

/// @brief Dilation step after region growing
/// @param context  Region growing context
/// @param points   Point cloud
/// @param search   Nearest neighbor search
/// @param params   Region growing parameters
template <typename PointCloud>
void region_growing_dilation_(
  RegionGrowingContext& context,
  const PointCloud& points,
  const NearestNeighborSearch& search,
  const RegionGrowingParams& params);

/// @brief Update region growing several steps
/// @param context  Region growing context
/// @param points   Point cloud
/// @param search   Nearest neighbor search
/// @param params   Region growing parameters
/// @return         True if the region growing is converged
template <typename PointCloud>
bool region_growing_update_(
  RegionGrowingContext& context,
  const PointCloud& points,
  const NearestNeighborSearch& search,
  const RegionGrowingParams& params);

RegionGrowingContext region_growing_init(
  const PointCloud& points,
  const NearestNeighborSearch& search,
  const Eigen::Vector4d& seed_point,
  const RegionGrowingParams& params);

bool region_growing_update(
  RegionGrowingContext& context,
  const PointCloud& points,
  const NearestNeighborSearch& search,
  const RegionGrowingParams& params);

}  // namespace gtsam_points