// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#include <gtsam_points/ann/kdtree_gpu.hpp>

#include <algorithm>
#include <thrust/copy.h>

#include <thrust/pair.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <gtsam_points/cuda/check_error.cuh>
#include <gtsam_points/cuda/kernels/kdtree.cuh>

namespace gtsam_points {

namespace {

struct nn_search_kernel {
public:
  __device__ void operator()(std::uint32_t i) const {
    const Eigen::Vector3f query = queries[i];

    kdtree_nearest_neighbor_search_kernel search{points, indices, nodes};
    const auto result = search(query);

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
    nn_search_kernel{points->points_gpu, indices, nodes, queries, nn_indices, nn_sq_dists});
}

}  // namespace gtsam_points