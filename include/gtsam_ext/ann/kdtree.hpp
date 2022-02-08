// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <iostream>
#include <Eigen/Core>

#include <gtsam_ext/ann/nearest_neighbor_search.hpp>

// forward declaration
namespace nanoflann {

template <class T, class DataSource, typename _DistanceType>
class L2_Simple_Adaptor;

template <typename Distance, class DatasetAdaptor, int DIM, typename IndexType>
class KDTreeSingleIndexAdaptor;

}  // namespace nanoflann

namespace gtsam_ext {

/**
 * @brief Wrapper for nanoflann
 */
struct KdTree : public NearestNeighborSearch {
public:
  using Index = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, KdTree, double>, KdTree, 3, size_t>;

  KdTree(const Eigen::Vector4d* points, int num_points);
  virtual ~KdTree() override;

  inline size_t kdtree_get_point_count() const { return num_points; }
  inline double kdtree_get_pt(const size_t idx, const size_t dim) const { return points[idx][dim]; }

  template <class BBox>
  bool kdtree_get_bbox(BBox&) const {
    return false;
  }

  virtual size_t knn_search(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists) const override;

public:
  const int num_points;
  const Eigen::Vector4d* points;

  double search_eps;

  std::unique_ptr<Index> index;
};

}  // namespace gtsam_ext
