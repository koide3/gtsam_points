// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <Eigen/Core>
#include <gtsam_points/ann/nearest_neighbor_search.hpp>

// forward declaration
namespace nanoflann {

template <class T, class DataSource, typename _DistanceType>
class L2_Simple_Adaptor;

template <typename Distance, class DatasetAdaptor, int DIM, typename IndexType>
class KDTreeSingleIndexAdaptor;

}  // namespace nanoflann

namespace gtsam_points {

/**
 * @brief KdTree with arbitrary dimension
 */
template <int D = -1>
struct KdTreeX : public NearestNeighborSearch {
public:
  using Index = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, KdTreeX<D>, double>, KdTreeX<D>, D, size_t>;

  /**
   * @brief Constructor
   */
  KdTreeX(const Eigen::Matrix<double, D, 1>* points, int num_points);
  virtual ~KdTreeX() override;

  int dim() const { return D != -1 ? D : points[0].size(); }
  inline size_t kdtree_get_point_count() const { return num_points; }
  inline double kdtree_get_pt(const size_t idx, const size_t dim) const { return points[idx][dim]; }

  template <class BBox>
  bool kdtree_get_bbox(BBox&) const {
    return false;
  }

  virtual size_t knn_search(
    const double* pt,
    size_t k,
    size_t* k_indices,
    double* k_sq_dists,
    double max_sq_dist = std::numeric_limits<double>::max()) const override;

  virtual size_t radius_search(
    const double* pt,
    double radius,
    std::vector<size_t>& indices,
    std::vector<double>& sq_dists,
    int max_num_neighbors = std::numeric_limits<int>::max()) const override;

public:
  const int num_points;
  const Eigen::Matrix<double, D, 1>* points;

  double search_eps;
  std::unique_ptr<Index> index;
};

}  // namespace gtsam_points