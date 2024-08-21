// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#include <gtsam_points/ann/intensity_kdtree.hpp>

#include <nanoflann.hpp>

namespace gtsam_points {

IntensityKdTree::IntensityKdTree(const Eigen::Vector4d* points, const double* intensities, int num_points, double intensity_scale)
: num_points(num_points),
  points(points),
  intensities(intensities),
  intensity_scale(intensity_scale),
  search_eps(-1.0),
  index(new Index(4, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10))) {
  index->buildIndex();
}

IntensityKdTree::~IntensityKdTree() {}

size_t IntensityKdTree::knn_search(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists, double max_sq_dist) const {
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
}