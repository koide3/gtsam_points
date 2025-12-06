// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#include <gtsam_points/ann/kdtree_cuda.hpp>

#include <algorithm>
#include <thrust/copy.h>

#include <thrust/pair.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <gtsam_points/cuda/check_error.cuh>

namespace gtsam_points {

namespace {

struct nearest_neighbor_search_kernel {
public:
  static constexpr int MAX_STACK_SIZE = 20;

  __device__ void operator()(std::uint32_t i) const {
    const Eigen::Vector3f query = queries[i];

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

    nn_indices[i] = result.first;
    nn_sq_dists[i] = result.second;
  }

public:
  const Eigen::Vector3f* __restrict__ points;
  const std::uint32_t* __restrict__ indices;
  const KdTreeNodeGPU* __restrict__ nodes;

  const Eigen::Vector3f* __restrict__ queries;

  std::uint32_t* nn_indices;
  float* nn_sq_dists;
};

}  // namespace

void KdTreeGPU::nearest_neighbor_search(
  const Eigen::Vector3f* queries,
  const size_t num_queries,
  std::uint32_t* nn_indices,
  float* nn_sq_dists,
  CUstream_st* stream) {
  thrust::for_each(
    thrust::cuda::par.on(stream),
    thrust::counting_iterator<std::uint32_t>(0),
    thrust::counting_iterator<std::uint32_t>(num_queries),
    nearest_neighbor_search_kernel{points->points_gpu, indices, nodes, queries, nn_indices, nn_sq_dists});
}

}  // namespace gtsam_points