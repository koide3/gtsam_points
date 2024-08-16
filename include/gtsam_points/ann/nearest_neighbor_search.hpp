// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <vector>
#include <limits>

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
   * @param k_sq_dists  Squared distances to the neighbors
   */
  virtual size_t
  knn_search(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists, double max_sq_dist = std::numeric_limits<double>::max()) const {
    return 0;
  };
};
}  // namespace gtsam_points