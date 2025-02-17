// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#include <gtsam_points/ann/kdtree.hpp>

#include <gtsam_points/config.hpp>
#include <gtsam_points/types/frame_traits.hpp>
#include <gtsam_points/ann/small_kdtree.hpp>
#include <gtsam_points/util/parallelism.hpp>

namespace gtsam_points {

namespace frame {

template <>
struct traits<KdTree> {
  static int size(const KdTree& tree) { return tree.num_points; }
  static bool has_points(const KdTree& tree) { return tree.num_points; }
  static const Eigen::Vector4d& point(const KdTree& tree, size_t i) { return tree.points[i]; }
};

}  // namespace frame

KdTree::KdTree(const Eigen::Vector4d* points, int num_points, int build_num_threads)
: num_points(num_points),
  points(points),
  search_eps(-1.0),
  index(
    is_omp_default() || build_num_threads == 1 ?             //
      new Index(*this, KdTreeBuilderOMP(build_num_threads))  //
                                               :             //
#ifdef GTSAM_POINTS_USE_TBB                                  //
      new Index(*this, KdTreeBuilderTBB())                   //
#else                                                        //
      new Index(*this, KdTreeBuilder())
#endif
  ) {
}

KdTree::~KdTree() {}

size_t KdTree::knn_search(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists, double max_sq_dist) const {
  KnnSetting setting;
  setting.max_sq_dist = max_sq_dist;
  if (k == 1) {
    return index->nearest_neighbor_search(Eigen::Map<const Eigen::Vector3d>(pt), k_indices, k_sq_dists, setting);
  } else {
    return index->knn_search(Eigen::Map<const Eigen::Vector3d>(pt), k, k_indices, k_sq_dists, setting);
  }
}

size_t KdTree::radius_search(const double* pt, double radius, std::vector<size_t>& indices, std::vector<double>& sq_dists, int max_num_neighbors)
  const {
  KnnSetting setting;
  setting.max_nn = max_num_neighbors;
  return index->radius_search(Eigen::Map<const Eigen::Vector3d>(pt), radius, indices, sq_dists, setting);
}

};  // namespace gtsam_points