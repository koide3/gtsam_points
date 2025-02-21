// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <vector>
#include <limits>
#include <iostream>

namespace gtsam_points {

/**
 * @brief Nearest neighbor search interface
 */
struct NearestNeighborSearch {
public:
  using Ptr = std::shared_ptr<NearestNeighborSearch>;
  using ConstPtr = std::shared_ptr<const NearestNeighborSearch>;

  NearestNeighborSearch() {}
  virtual ~NearestNeighborSearch() {}

  /**
   * @brief k-nearest neighbor search
   * @param pt          Point
   * @param k           Number of neighbors
   * @param k_indices   Indices of k-nearest neighbors
   * @param k_sq_dists  Squared distances to the neighbors (sorted in ascending order)
   */
  virtual size_t
  knn_search(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists, double max_sq_dist = std::numeric_limits<double>::max()) const {
    std::cerr << "NearestNeighborSearch::knn_search() is not implemented" << std::endl;
    return 0;
  };

  /**
   * @brief Radius search
   * @note  There is no assumption and guarantee on the order of points to be selected when `max_num_neighbors` is specified.
   *        (Some algorithms like KdTree tend to first pick closer points though).
   * @param pt                 Point
   * @param radius             Search radius
   * @param indices            Indices of neighbors within the radius
   * @param sq_dists           Squared distances to the neighbors (sorted in ascending order)
   * @param max_num_neighbors  Maximum number of neighbors
   * @return                   Number of neighbors
   */
  virtual size_t radius_search(
    const double* pt,
    double radius,
    std::vector<size_t>& indices,
    std::vector<double>& sq_dists,
    int max_num_neighbors = std::numeric_limits<int>::max()) const {
    std::cerr << "NearestNeighborSearch::radius_search() is not implemented" << std::endl;
    return 0;
  };
};
}  // namespace gtsam_points