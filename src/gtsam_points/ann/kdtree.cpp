#include <gtsam_points/ann/kdtree.hpp>

#include <nanoflann.hpp>

namespace gtsam_points {

KdTree::KdTree(const Eigen::Vector4d* points, int num_points)
: num_points(num_points),
  points(points),
  search_eps(-1.0),
  index(new Index(3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10))) {
  index->buildIndex();
}

KdTree::~KdTree() {}

size_t KdTree::knn_search(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists) const {
  if (search_eps > 0.0) {
    nanoflann::KNNResultSet<double, size_t> result_set(k);
    result_set.init(k_indices, k_sq_dists);
    nanoflann::SearchParams search_params;
    search_params.eps = search_eps;
    index->findNeighbors(result_set, pt, search_params);
    return result_set.size();
  }

  return index->knnSearch(pt, k, k_indices, k_sq_dists);
}

};  // namespace gtsam_points