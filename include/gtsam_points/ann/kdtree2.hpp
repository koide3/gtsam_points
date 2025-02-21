// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <iostream>
#include <Eigen/Core>

#include <gtsam_points/config.hpp>
#include <gtsam_points/ann/small_kdtree.hpp>
#include <gtsam_points/types/frame_traits.hpp>
#include <gtsam_points/ann/nearest_neighbor_search.hpp>
#include <gtsam_points/util/parallelism.hpp>

namespace gtsam_points {

/**
 * @brief KdTree-based nearest neighbor search
 */
template <typename Frame>
struct KdTree2 : public NearestNeighborSearch {
public:
  using Index = UnsafeKdTree<Frame>;

  KdTree2(const std::shared_ptr<const Frame>& frame, int build_num_threads = 1)
  : frame(frame),
    search_eps(-1.0),
    index(
      is_omp_default() || build_num_threads == 1 ?                    //
        new Index(*this->frame, KdTreeBuilderOMP(build_num_threads))  //
                                                 :                    //
#ifdef GTSAM_POINTS_USE_TBB                                           //
        new Index(*this->frame, KdTreeBuilderTBB())                   //
#else                                                                 //
        new Index(*this->frame, KdTreeBuilder())
#endif
    ) {
    if (frame::size(*frame) == 0) {
      std::cerr << "error: empty frame is given for KdTree2" << std::endl;
      std::cerr << "     : frame::size() may not be implemented" << std::endl;
    }
  }
  virtual ~KdTree2() override {}

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
    double max_sq_dist = std::numeric_limits<double>::max()) const override {
    KnnSetting setting;
    setting.max_sq_dist = max_sq_dist;
    return index->knn_search(Eigen::Map<const Eigen::Vector3d>(pt), k, k_indices, k_sq_dists, setting);
  }

  /**
   * @brief Radius search
   * @note  There is no assumption and guarantee on the order of points to be selected when `max_num_neighbors` is specified.
   *        (KdTree tends to first pick closer points though).
   * @param pt                 Point
   * @param radius             Search radius
   * @param indices            Indices of neighbors within the radius
   * @param sq_dists           Squared distances to the neighbors
   * @param max_num_neighbors  Maximum number of neighbors
   * @return                   Number of neighbors
   */
  virtual size_t radius_search(
    const double* pt,
    double radius,
    std::vector<size_t>& indices,
    std::vector<double>& sq_dists,
    int max_num_neighbors = std::numeric_limits<int>::max()) const override {
    KnnSetting setting;
    setting.max_nn = max_num_neighbors;
    return index->radius_search(Eigen::Map<const Eigen::Vector3d>(pt), radius, indices, sq_dists, setting);
  }

public:
  const std::shared_ptr<const Frame> frame;

  double search_eps;

  std::unique_ptr<Index> index;
};

}  // namespace gtsam_points
