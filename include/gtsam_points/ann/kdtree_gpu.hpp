// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#pragma once

#include <limits>
#include <cstdint>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/ann/small_kdtree.hpp>

struct CUstream_st;

namespace gtsam_points {

struct KdTreeNodeGPU {
  NodeIndexType left = INVALID_NODE;   ///< Left child node index.
  NodeIndexType right = INVALID_NODE;  ///< Right child node index.

  union {
    struct Leaf {
      NodeIndexType first;  ///< First point index in the leaf node.
      NodeIndexType last;   ///< Last point index in the leaf node.
    } lr;                   ///< Leaf node.
    struct NonLeaf {
      NodeIndexType axis;  ///< Projection axis.
      float thresh;        ///< Threshold value.
    } sub;                 ///< Non-leaf node.
  } node_type;
};

class KdTreeGPU {
public:
  using Ptr = std::shared_ptr<KdTreeGPU>;
  using ConstPtr = std::shared_ptr<const KdTreeGPU>;

  KdTreeGPU(const PointCloud::ConstPtr& points, CUstream_st* stream = nullptr);
  ~KdTreeGPU();

  void nearest_neighbor_search(
    const Eigen::Vector3f* queries,
    size_t num_queries,
    std::uint32_t* nn_indices,
    float* nn_sq_dists,
    CUstream_st* stream = nullptr);

  void nearest_neighbor_search_cpu(
    const Eigen::Vector3f* h_queries,
    size_t num_queries,
    std::uint32_t* h_nn_indices,
    float* h_nn_sq_dists,
    CUstream_st* stream = nullptr);

  /// @brief Get the GPU pointer to the point indices
  const std::uint32_t* get_indices() const { return indices; }

  /// @brief Get the GPU pointer to the KdTree nodes
  const KdTreeNodeGPU* get_nodes() const { return nodes; }

private:
  PointCloud::ConstPtr points;
  size_t num_indices;
  size_t num_nodes;
  std::uint32_t* indices;
  KdTreeNodeGPU* nodes;
};

}  // namespace gtsam_points
