#pragma once

#include <Eigen/Core>
#include <nanoflann.hpp>

namespace gtsam_ext {

struct KdTree {
public:
  using Index = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, KdTree>, KdTree, 3>;

  KdTree(int num_points, const Eigen::Vector4d* points) : num_points(num_points), points(points), index(3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10)) {
    index.buildIndex();
  }

  inline size_t kdtree_get_point_count() const { return num_points; }
  inline double kdtree_get_pt(const size_t idx, const size_t dim) const { return points[idx][dim]; }

  template <class BBox>
  bool kdtree_get_bbox(BBox&) const {
    return false;
  }

public:
  const int num_points;
  const Eigen::Vector4d* points;

  Index index;
};
}  // namespace gtsam_ext
