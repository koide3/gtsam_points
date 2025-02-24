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

/// @brief Parameters for min-cut segmentation
struct MinCutParams {
  double distance_sigma = 0.25;              ///< Distance sigma
  double angle_sigma = 10.0 * M_PI / 180.0;  ///< Angle sigma

  double foreground_mask_radius = 0.2;  ///< All points within this radius from the source point are considered as foreground
  double background_mask_radius = 3.0;  ///< All points out of this radius from the source point are considered as background
  double foreground_weight = 0.2;       ///< Weight for the foreground points
  double background_weight = 0.2;       ///< Weight for the background points

  int k_neighbors = 20;  ///< Number of neighbors
  int num_threads = 1;   ///< Number of threads
};

/// @brief Result of min-cut segmentation
struct MinCutResult {
  MinCutResult() : source_index(-1), sink_index(-1), max_flow(0.0) {}

  size_t source_index;                  ///< Source point index
  size_t sink_index;                    ///< Sink point index
  double max_flow;                      ///< Maximum flow
  std::vector<size_t> cluster_indices;  ///< Indices of foreground points
};

/// @brief Min-cut segmentation
/// @param points           Point cloud
/// @param search           Nearest neighbor search
/// @param source_pt_index  Index of the source point
/// @param params           Parameters
/// @return                 Segmentation result
template <typename PointCloud>
MinCutResult min_cut_(const PointCloud& points, const NearestNeighborSearch& search, size_t source_pt_index, const MinCutParams& params);

/// @brief Min-cut segmentation
/// @param points       Point cloud
/// @param search       Nearest neighbor search
/// @param source_pt    Source point (The point nearest to this point is used as the source point)
/// @param params       Parameters
/// @return             Segmentation result
template <typename PointCloud>
MinCutResult min_cut_(const PointCloud& points, const NearestNeighborSearch& search, const Eigen::Vector4d& source_pt, const MinCutParams& params);

MinCutResult min_cut(const PointCloud& points, const NearestNeighborSearch& search, size_t source_pt_index, const MinCutParams& params);

MinCutResult min_cut(const PointCloud& points, const NearestNeighborSearch& search, const Eigen::Vector4d& source_pt, const MinCutParams& params);

}  // namespace gtsam_points