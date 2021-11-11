#pragma once

#include <gtsam_ext/ann/kdtree.hpp>

namespace gtsam_ext {

struct IntensityKdTree : public NearestNeighborSearch {
public:
  using Index = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, IntensityKdTree>, IntensityKdTree, 4>;

  IntensityKdTree(const Eigen::Vector4d* points, const double* intensities, int num_points, double intensity_scale = 1.0)
  : num_points(num_points),
    points(points),
    intensities(intensities),
    intensity_scale(intensity_scale),
    search_eps(-1.0),
    index(4, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10)) {
    index.buildIndex();
  }
  virtual ~IntensityKdTree() override {}

  inline size_t kdtree_get_point_count() const { return num_points; }
  inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
    if (dim < 3) {
      return points[idx][dim];
    } else {
      return intensities[idx] * intensity_scale;
    }
  }

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
  const double* intensities;
  const Eigen::Vector4d* points;
  const double intensity_scale;

  double search_eps;

  Index index;
};

}  // namespace gtsam_ext