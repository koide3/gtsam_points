// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#include <gtsam_points/ann/kdtree_gpu.hpp>

namespace gtsam_points {

struct kdtree_nearest_neighbor_search_kernel {
public:
  static constexpr int MAX_STACK_SIZE = 20;

  __device__ thrust::pair<NodeIndexType, float> operator()(const Eigen::Vector3f query) const {
    thrust::pair<NodeIndexType, float> result = {INVALID_NODE, std::numeric_limits<float>::max()};

    int stack_size = 1;
    thrust::pair<int, float> search_stack[MAX_STACK_SIZE] = {{0, 0.0f}};

    while (stack_size > 0) {
      const auto [node_index, sq_dist] = search_stack[--stack_size];
      if (sq_dist > result.second) {
        continue;
      }

      const KdTreeNodeGPU node = nodes[node_index];

      // Leaf node
      if (node.left == INVALID_NODE) {
        for (NodeIndexType i = node.node_type.lr.first; i < node.node_type.lr.last; i++) {
          const NodeIndexType pt_index = indices[i];
          const float sq_dist = (points[pt_index] - query).squaredNorm();
          if (sq_dist < result.second) {
            result = {pt_index, sq_dist};
          }
        }
        continue;
      }

      const float val = query[node.node_type.sub.axis];
      const float diff = val - node.node_type.sub.thresh;
      const float cut_sq_dist = diff * diff;

      int best_child;
      int other_child;

      if (diff < 0.0f) {
        best_child = node.left;
        other_child = node.right;
      } else {
        best_child = node.right;
        other_child = node.left;
      }

      if (stack_size > MAX_STACK_SIZE - 2) {
        printf("kdtree stack overflow!!");
      } else if (cut_sq_dist < result.second) {
        search_stack[stack_size].first = other_child;
        search_stack[stack_size++].second = cut_sq_dist;
      }

      search_stack[stack_size].first = best_child;
      search_stack[stack_size++].second = 0.0f;
    }

    return result;
  }

public:
  const Eigen::Vector3f* __restrict__ points;
  const std::uint32_t* __restrict__ indices;
  const KdTreeNodeGPU* __restrict__ nodes;
};

}  // namespace gtsam_points
