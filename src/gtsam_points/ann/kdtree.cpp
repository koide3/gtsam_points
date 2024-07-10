// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#include <gtsam_points/ann/kdtree.hpp>

#include <gtsam_points/types/frame_traits.hpp>
#include <gtsam_points/ann/small_kdtree.hpp>

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
  index(new Index(*this, KdTreeBuilderOMP(build_num_threads))) {}

KdTree::~KdTree() {}

size_t KdTree::knn_search(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists) const {
  return index->knn_search(Eigen::Map<const Eigen::Vector3d>(pt), k, k_indices, k_sq_dists);
}

};  // namespace gtsam_points