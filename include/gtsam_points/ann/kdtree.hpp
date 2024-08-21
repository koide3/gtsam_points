// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <iostream>
#include <Eigen/Core>

#include <gtsam_points/ann/nearest_neighbor_search.hpp>

namespace gtsam_points {

template <typename PointCloud, typename Projection>
struct UnsafeKdTree;
struct AxisAlignedProjection;

/**
 * @brief KdTree-based nearest neighbor search
 */
struct KdTree : public NearestNeighborSearch {
public:
  using Index = UnsafeKdTree<KdTree, AxisAlignedProjection>;

  KdTree(const Eigen::Vector4d* points, int num_points, int build_num_threads = 1);
  virtual ~KdTree() override;

  /// @brief Find k nearest neighbors
  /// @param pt           Query point (must be 4D vector [x, y, z, 1])
  /// @param k            Number of neighbors to search
  /// @param k_indices    Indices of k nearest neighbors
  /// @param k_sq_dists   Squared distances of k nearest neighbors
  /// @return             Number of neighbors found
  virtual size_t knn_search(
    const double* pt,
    size_t k,
    size_t* k_indices,
    double* k_sq_dists,
    double max_sq_dist = std::numeric_limits<double>::max()) const override;

public:
  const int num_points;
  const Eigen::Vector4d* points;

  double search_eps;

  std::unique_ptr<Index> index;
};

}  // namespace gtsam_points
