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

struct MinCutParams {
  double distance_sigma = 0.25;              ///< Distance sigma
  double angle_sigma = 10.0 * M_PI / 180.0;  ///< Angle sigma

  double foreground_mask_radius = 0.2;  ///< All points within this radius from the source point are considered as foreground
  double background_mask_radius = 3.0;  ///< All points out of this radius from the source point are considered as background
  double foreground_weight = 0.2;       ///< Weight for the foreground points
  double background_weight = 0.2;       ///< Weight for the background points

  int k_neighbors = 10;  ///< Number of neighbors
  int num_threads = 1;   ///< Number of threads
};

struct MinCutResult {
  double max_flow;
  std::vector<size_t> cluster_indices;
};

template <typename PointCloud>
MinCutResult
min_cut_(const PointCloud& points, const NearestNeighborSearch& search, size_t source_pt_index, size_t sink_pt_index, const MinCutParams& params);

}  // namespace gtsam_points