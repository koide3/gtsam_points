// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/ann/kdtreex.hpp>

#include <nanoflann.hpp>

namespace gtsam_points {

template <int D>
KdTreeX<D>::KdTreeX(const Eigen::Matrix<double, D, 1>* points, int num_points)
: num_points(num_points),
  points(points),
  search_eps(-1.0),
  index(new Index(dim(), *this, nanoflann::KDTreeSingleIndexAdaptorParams(10))) {
  index->buildIndex();
}

template <int D>
KdTreeX<D>::~KdTreeX() {}

template <int D>
size_t KdTreeX<D>::knn_search(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists, double max_sq_dist) const {
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

template <int D>
size_t KdTreeX<D>::radius_search(const double* pt, double radius, std::vector<size_t>& indices, std::vector<double>& sq_dists, int max_num_neighbors)
  const {
  nanoflann::SearchParams params;
  std::vector<std::pair<size_t, double>> indices_dists;
  size_t num_found = index->radiusSearch(pt, radius * radius, indices_dists, params);

  indices.resize(num_found);
  sq_dists.resize(num_found);
  for (size_t i = 0; i < num_found; i++) {
    indices[i] = indices_dists[i].first;
    sq_dists[i] = indices_dists[i].second;
  }

  return num_found;
}

}  // namespace gtsam_points
