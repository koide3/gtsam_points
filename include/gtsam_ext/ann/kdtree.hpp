#pragma once

#include <iostream>
#include <Eigen/Core>
#include <nanoflann.hpp>

#include <gtsam_ext/ann/nearest_neighbor_search.hpp>

namespace gtsam_ext {

struct KdTree : public NearestNeighborSearch {
public:
  using Index = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, KdTree>, KdTree, 3>;

  KdTree(const Eigen::Vector4d* points, int num_points) : num_points(num_points), points(points), search_eps(-1.0), index(3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10)) {
    index.buildIndex();
  }
  virtual ~KdTree() override {}

  inline size_t kdtree_get_point_count() const { return num_points; }
  inline double kdtree_get_pt(const size_t idx, const size_t dim) const { return points[idx][dim]; }

  template <class BBox>
  bool kdtree_get_bbox(BBox&) const {
    return false;
  }

  virtual size_t knn_search(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists) const override {
    if (search_eps > 0.0) {
      nanoflann::KNNResultSet<double, size_t> result_set(k);
      result_set.init(k_indices, k_sq_dists);
      nanoflann::SearchParams search_params;
      search_params.eps = search_eps;
      index.findNeighbors(result_set, pt, search_params);
      return result_set.size();
    }

    return index.knnSearch(pt, k, k_indices, k_sq_dists);
  }

public:
  const int num_points;
  const Eigen::Vector4d* points;

  double search_eps;

  Index index;
};

}  // namespace gtsam_ext
