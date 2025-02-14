// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT

// KdTree code derived from small_gicp.
// While the following KdTree code is written from scratch, it is heavily inspired by the nanoflann library.
// Thus, the following original license of nanoflann is included to be sure.

// https://github.com/jlblancoc/nanoflann/blob/master/include/nanoflann.hpp
/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 * Copyright 2011-2024  Jose Luis Blanco (joseluisblancoc@gmail.com).
 *   All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/
#pragma once

#include <atomic>
#include <memory>
#include <numeric>
#include <Eigen/Core>

#include <gtsam_points/config.hpp>
#include <gtsam_points/ann/knn_result.hpp>
#include <gtsam_points/types/frame_traits.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/parallel_invoke.h>
#endif

namespace gtsam_points {

/// @brief Parameters to control the projection axis search.
struct ProjectionSetting {
  int max_scan_count = 128;  ///< Maximum number of points to use for the axis search.
};

/// @brief Conventional axis-aligned projection (i.e., selecting any of XYZ axes with the largest variance).
struct AxisAlignedProjection {
public:
  /// @brief Project the point to the selected axis.
  /// @param pt  Point to project
  /// @return    Projected value
  double operator()(const Eigen::Vector4d& pt) const { return pt[axis]; }

  /// @brief Find the axis with the largest variance.
  /// @param points     Point cloud
  /// @param first      First point index iterator
  /// @param last       Last point index iterator
  /// @param setting    Search setting
  /// @return           Projection with the largest variance axis
  template <typename PointCloud, typename IndexConstIterator>
  static AxisAlignedProjection
  find_axis(const PointCloud& points, IndexConstIterator first, IndexConstIterator last, const ProjectionSetting& setting) {
    const size_t N = std::distance(first, last);
    Eigen::Vector4d sum_pt = Eigen::Vector4d::Zero();
    Eigen::Vector4d sum_sq = Eigen::Vector4d::Zero();

    const size_t step = N < setting.max_scan_count ? 1 : N / setting.max_scan_count;
    const size_t num_steps = N / step;
    for (int i = 0; i < num_steps; i++) {
      const auto itr = first + step * i;
      const Eigen::Vector4d pt = frame::point(points, *itr);
      sum_pt += pt;
      sum_sq += pt.cwiseProduct(pt);
    }

    const Eigen::Vector4d mean = sum_pt / sum_pt.w();
    const Eigen::Vector4d var = (sum_sq - mean.cwiseProduct(sum_pt));

    return AxisAlignedProjection{var[0] > var[1] ? (var[0] > var[2] ? 0 : 2) : (var[1] > var[2] ? 1 : 2)};
  }

public:
  int axis;  ///< Axis index (0: X, 1: Y, 2: Z)
};

using NodeIndexType = std::uint32_t;
static constexpr NodeIndexType INVALID_NODE = std::numeric_limits<NodeIndexType>::max();

/// @brief KdTree node.
template <typename Projection>
struct KdTreeNode {
  union {
    struct Leaf {
      NodeIndexType first;  ///< First point index in the leaf node.
      NodeIndexType last;   ///< Last point index in the leaf node.
    } lr;                   ///< Leaf node.
    struct NonLeaf {
      Projection proj;  ///< Projection axis.
      double thresh;    ///< Threshold value.
    } sub;              ///< Non-leaf node.
  } node_type;

  NodeIndexType left = INVALID_NODE;   ///< Left child node index.
  NodeIndexType right = INVALID_NODE;  ///< Right child node index.
};

/// @brief Single thread Kd-tree builder.
struct KdTreeBuilder {
public:
  /// @brief Build KdTree
  /// @param kdtree         Kd-tree to build
  /// @param points         Point cloud
  template <typename KdTree, typename PointCloud>
  void build_tree(KdTree& kdtree, const PointCloud& points) const {
    kdtree.indices.resize(frame::size(points));
    std::iota(kdtree.indices.begin(), kdtree.indices.end(), 0);

    size_t node_count = 0;
    kdtree.nodes.resize(frame::size(points));
    kdtree.root = create_node(kdtree, node_count, points, kdtree.indices.begin(), kdtree.indices.begin(), kdtree.indices.end());
    kdtree.nodes.resize(node_count);
  }

  /// @brief Create a Kd-tree node from the given point indices.
  /// @param global_first     Global first point index iterator (i.e., this->indices.begin()).
  /// @param first            First point index iterator to be scanned.
  /// @param last             Last point index iterator to be scanned.
  /// @return                 Index of the created node.
  template <typename PointCloud, typename KdTree, typename IndexConstIterator>
  NodeIndexType create_node(
    KdTree& kdtree,
    size_t& node_count,
    const PointCloud& points,
    IndexConstIterator global_first,
    IndexConstIterator first,
    IndexConstIterator last) const {
    const size_t N = std::distance(first, last);
    const NodeIndexType node_index = node_count++;
    auto& node = kdtree.nodes[node_index];

    // Create a leaf node.
    if (N <= max_leaf_size) {
      // std::sort(first, last);
      node.node_type.lr.first = std::distance(global_first, first);
      node.node_type.lr.last = std::distance(global_first, last);

      return node_index;
    }

    // Find the best axis to split the input points.
    using Projection = typename KdTree::Projection;
    const auto proj = Projection::find_axis(points, first, last, projection_setting);
    const auto median_itr = first + N / 2;
    std::nth_element(first, median_itr, last, [&](size_t i, size_t j) { return proj(frame::point(points, i)) < proj(frame::point(points, j)); });

    // Create a non-leaf node.
    node.node_type.sub.proj = proj;
    node.node_type.sub.thresh = proj(frame::point(points, *median_itr));

    // Create left and right child nodes.
    node.left = create_node(kdtree, node_count, points, global_first, first, median_itr);
    node.right = create_node(kdtree, node_count, points, global_first, median_itr, last);

    return node_index;
  }

public:
  int max_leaf_size = 20;                ///< Maximum number of points in a leaf node.
  ProjectionSetting projection_setting;  ///< Projection setting.
};

/// @brief Kd-tree builder with OpenMP.
struct KdTreeBuilderOMP {
public:
  /// @brief Constructor
  /// @param num_threads  Number of threads
  KdTreeBuilderOMP(int num_threads = 4) : num_threads(num_threads), max_leaf_size(20) {}

  /// @brief Build KdTree
  template <typename KdTree, typename PointCloud>
  void build_tree(KdTree& kdtree, const PointCloud& points) const {
    kdtree.indices.resize(frame::size(points));
    std::iota(kdtree.indices.begin(), kdtree.indices.end(), 0);

    std::atomic_uint64_t node_count = 0;
    kdtree.nodes.resize(frame::size(points));

#ifndef _MSC_VER
#pragma omp parallel num_threads(num_threads)
    {
#pragma omp single nowait
      {
        kdtree.root = create_node(kdtree, node_count, points, kdtree.indices.begin(), kdtree.indices.begin(), kdtree.indices.end());
      }
    }
#else
    kdtree.root = create_node(kdtree, node_count, points, kdtree.indices.begin(), kdtree.indices.begin(), kdtree.indices.end());
#endif

    kdtree.nodes.resize(node_count);
  }

  /// @brief Create a Kd-tree node from the given point indices.
  /// @param global_first     Global first point index iterator (i.e., this->indices.begin()).
  /// @param first            First point index iterator to be scanned.
  /// @param last             Last point index iterator to be scanned.
  /// @return                 Index of the created node.
  template <typename PointCloud, typename KdTree, typename IndexConstIterator>
  NodeIndexType create_node(
    KdTree& kdtree,
    std::atomic_uint64_t& node_count,
    const PointCloud& points,
    IndexConstIterator global_first,
    IndexConstIterator first,
    IndexConstIterator last) const {
    const size_t N = std::distance(first, last);
    const NodeIndexType node_index = node_count++;
    auto& node = kdtree.nodes[node_index];

    // Create a leaf node.
    if (N <= max_leaf_size) {
      // std::sort(first, last);
      node.node_type.lr.first = std::distance(global_first, first);
      node.node_type.lr.last = std::distance(global_first, last);

      return node_index;
    }

    // Find the best axis to split the input points.
    using Projection = typename KdTree::Projection;
    const auto proj = Projection::find_axis(points, first, last, projection_setting);
    const auto median_itr = first + N / 2;
    std::nth_element(first, median_itr, last, [&](size_t i, size_t j) { return proj(frame::point(points, i)) < proj(frame::point(points, j)); });

    // Create a non-leaf node.
    node.node_type.sub.proj = proj;
    node.node_type.sub.thresh = proj(frame::point(points, *median_itr));

    // Create left and right child nodes.
#ifndef _MSC_VER
#pragma omp task default(shared) if (N > 512)
    node.left = create_node(kdtree, node_count, points, global_first, first, median_itr);
#pragma omp task default(shared) if (N > 512)
    node.right = create_node(kdtree, node_count, points, global_first, median_itr, last);
#pragma omp taskwait
#else
    node.left = create_node(kdtree, node_count, points, global_first, first, median_itr);
    node.right = create_node(kdtree, node_count, points, global_first, median_itr, last);
#endif

    return node_index;
  }

public:
  int num_threads;                       ///< Number of threads
  int max_leaf_size;                     ///< Maximum number of points in a leaf node.
  ProjectionSetting projection_setting;  ///< Projection setting.
};

#ifdef GTSAM_POINTS_USE_TBB
/// @brief Kd-tree builder with TBB.
struct KdTreeBuilderTBB {
public:
  /// @brief Build KdTree
  template <typename KdTree, typename PointCloud>
  void build_tree(KdTree& kdtree, const PointCloud& points) const {
    kdtree.indices.resize(frame::size(points));
    std::iota(kdtree.indices.begin(), kdtree.indices.end(), 0);

    std::atomic_uint64_t node_count = 0;
    kdtree.nodes.resize(frame::size(points));
    kdtree.root = create_node(kdtree, node_count, points, kdtree.indices.begin(), kdtree.indices.begin(), kdtree.indices.end());
    kdtree.nodes.resize(node_count);
  }

  /// @brief Create a Kd-tree node from the given point indices.
  /// @param global_first     Global first point index iterator (i.e., this->indices.begin()).
  /// @param first            First point index iterator to be scanned.
  /// @param last             Last point index iterator to be scanned.
  /// @return                 Index of the created node.
  template <typename PointCloud, typename KdTree, typename IndexConstIterator>
  NodeIndexType create_node(
    KdTree& kdtree,
    std::atomic_uint64_t& node_count,
    const PointCloud& points,
    IndexConstIterator global_first,
    IndexConstIterator first,
    IndexConstIterator last) const {
    const size_t N = std::distance(first, last);
    const NodeIndexType node_index = node_count++;
    auto& node = kdtree.nodes[node_index];

    // Create a leaf node.
    if (N <= max_leaf_size) {
      // std::sort(first, last);
      node.node_type.lr.first = std::distance(global_first, first);
      node.node_type.lr.last = std::distance(global_first, last);

      return node_index;
    }

    // Find the best axis to split the input points.
    using Projection = typename KdTree::Projection;
    const auto proj = Projection::find_axis(points, first, last, projection_setting);
    const auto median_itr = first + N / 2;
    std::nth_element(first, median_itr, last, [&](size_t i, size_t j) { return proj(frame::point(points, i)) < proj(frame::point(points, j)); });

    // Create a non-leaf node.
    node.node_type.sub.proj = proj;
    node.node_type.sub.thresh = proj(frame::point(points, *median_itr));

    // Create left and right child nodes.
    if (N > 512) {
      tbb::parallel_invoke(
        [&] { node.left = create_node(kdtree, node_count, points, global_first, first, median_itr); },
        [&] { node.right = create_node(kdtree, node_count, points, global_first, median_itr, last); });
    } else {
      node.left = create_node(kdtree, node_count, points, global_first, first, median_itr);
      node.right = create_node(kdtree, node_count, points, global_first, median_itr, last);
    }

    return node_index;
  }

public:
  int max_leaf_size = 20;                ///< Maximum number of points in a leaf node.
  ProjectionSetting projection_setting;  ///< Projection setting.
};
#endif

/// @brief "Unsafe" KdTree.
/// @note  This class does not hold the ownership of the input points.
///        You must keep the input points along with this class.
template <typename PointCloud, typename Projection_ = AxisAlignedProjection>
struct UnsafeKdTree {
public:
  using Projection = Projection_;
  using Node = KdTreeNode<Projection>;

  /// @brief Constructor
  /// @param points   Point cloud
  /// @param builder  Kd-tree builder
  template <typename Builder = KdTreeBuilder>
  explicit UnsafeKdTree(const PointCloud& points, const Builder& builder = KdTreeBuilder()) : points(points) {
    if (frame::size(points) == 0) {
      std::cerr << "warning: Empty point cloud" << std::endl;
      return;
    }

    builder.build_tree(*this, points);
  }

  /// @brief Find the nearest neighbor.
  /// @param query        Query point
  /// @param k_indices    Index of the nearest neighbor
  /// @param k_sq_dists   Squared distance to the nearest neighbor
  /// @param setting      KNN search setting
  /// @return             Number of found neighbors (0 or 1)
  size_t nearest_neighbor_search(const Eigen::Vector3d& query_, size_t* k_indices, double* k_sq_dists, const KnnSetting& setting = KnnSetting())
    const {
    return knn_search<1>(query_, k_indices, k_sq_dists, setting);
  }

  /// @brief  Find k-nearest neighbors. This method uses dynamic memory allocation.
  /// @param  query       Query point
  /// @param  k           Number of neighbors
  /// @param  k_indices   Indices of neighbors
  /// @param  k_sq_dists  Squared distances to neighbors
  /// @param  setting     KNN search setting
  /// @return             Number of found neighbors
  size_t knn_search(const Eigen::Vector3d& query_, int k, size_t* k_indices, double* k_sq_dists, const KnnSetting& setting = KnnSetting()) const {
    const Eigen::Vector4d query = (Eigen::Vector4d() << query_, 1.0).finished();
    KnnResult<-1> result(k_indices, k_sq_dists, k, identity_transform(), setting.max_sq_dist);
    knn_search(query, root, result, setting);
    return result.num_found();
  }

  /// @brief Find k-nearest neighbors. This method uses fixed and static memory allocation. Might be faster for small k.
  /// @param query       Query point
  /// @param k_indices   Indices of neighbors
  /// @param k_sq_dists  Squared distances to neighbors
  /// @param setting     KNN search setting
  /// @return            Number of found neighbors
  template <int N>
  size_t knn_search(const Eigen::Vector3d& query_, size_t* k_indices, double* k_sq_dists, const KnnSetting& setting = KnnSetting()) const {
    const Eigen::Vector4d query = (Eigen::Vector4d() << query_, 1.0).finished();
    KnnResult<N> result(k_indices, k_sq_dists, -1, identity_transform(), setting.max_sq_dist);
    knn_search(query, root, result, setting);
    return result.num_found();
  }

  /// @brief Find neighbors in a search radius.
  /// @param query        Query point
  /// @param radius       Search radius
  /// @param indices      Indices of neighbors
  /// @param sq_dists     Squared distances to neighbors
  /// @param setting      KNN search setting
  /// @return             Number of found neighbors
  size_t radius_search(
    const Eigen::Vector3d& query_,
    double radius,
    std::vector<size_t>& indices,
    std::vector<double>& sq_dists,
    const KnnSetting& setting = KnnSetting()) const {
    const Eigen::Vector4d query = (Eigen::Vector4d() << query_, 1.0).finished();
    RadiusSearchResult result;
    radius_search(query, root, result, radius * radius, setting);
    result.sort();

    indices.resize(result.num_found());
    std::transform(result.neighbors.begin(), result.neighbors.end(), indices.begin(), [](const auto& p) { return p.first; });
    sq_dists.resize(result.num_found());
    std::transform(result.neighbors.begin(), result.neighbors.end(), sq_dists.begin(), [](const auto& p) { return p.second; });

    return result.num_found();
  }

private:
  /// @brief Find k-nearest neighbors.
  template <typename Result>
  bool knn_search(const Eigen::Vector4d& query, NodeIndexType node_index, Result& result, const KnnSetting& setting) const {
    const auto& node = nodes[node_index];

    // Check if it's a leaf node.
    if (node.left == INVALID_NODE) {
      // Compare the query point with all points in the leaf node.
      for (size_t i = node.node_type.lr.first; i < node.node_type.lr.last; i++) {
        const double sq_dist = (frame::point(points, indices[i]) - query).squaredNorm();
        result.push(indices[i], sq_dist);
      }
      return !setting.fulfilled(result);
    }

    const double val = node.node_type.sub.proj(query);
    const double diff = val - node.node_type.sub.thresh;
    const double cut_sq_dist = diff * diff;

    NodeIndexType best_child;
    NodeIndexType other_child;

    if (diff < 0.0) {
      best_child = node.left;
      other_child = node.right;
    } else {
      best_child = node.right;
      other_child = node.left;
    }

    // Check the best child node first.
    if (!knn_search(query, best_child, result, setting)) {
      return false;
    }

    // Check if the other child node needs to be tested.
    if (result.worst_distance() > cut_sq_dist) {
      return knn_search(query, other_child, result, setting);
    }

    return true;
  }

  /// @brief Find neighbors in a search radius.
  template <typename Result>
  bool radius_search(const Eigen::Vector4d& query, NodeIndexType node_index, Result& result, double sq_radius, const KnnSetting& setting) const {
    const auto& node = nodes[node_index];

    // Check if it's a leaf node.
    if (node.left == INVALID_NODE) {
      // Compare the query point with all points in the leaf node.
      for (size_t i = node.node_type.lr.first; i < node.node_type.lr.last; i++) {
        if (setting.fulfilled(result)) {
          return false;
        }

        const double sq_dist = (frame::point(points, indices[i]) - query).squaredNorm();
        if (sq_dist < sq_radius) {
          result.push(indices[i], sq_dist);
        }
      }
      return !setting.fulfilled(result);
    }

    const double val = node.node_type.sub.proj(query);
    const double diff = val - node.node_type.sub.thresh;
    const double cut_sq_dist = diff * diff;

    NodeIndexType best_child;
    NodeIndexType other_child;

    if (diff < 0.0) {
      best_child = node.left;
      other_child = node.right;
    } else {
      best_child = node.right;
      other_child = node.left;
    }

    // Check the best child node first.
    if (!radius_search(query, best_child, result, sq_radius, setting)) {
      return false;
    }

    // Check if the other child node needs to be tested.
    if (sq_radius > cut_sq_dist) {
      return radius_search(query, other_child, result, sq_radius, setting);
    }

    return true;
  }

public:
  const PointCloud& points;     ///< Input points
  std::vector<size_t> indices;  ///< Point indices refered by nodes

  NodeIndexType root;       ///< Root node index (should be zero)
  std::vector<Node> nodes;  ///< Kd-tree nodes
};

}  // namespace gtsam_points
