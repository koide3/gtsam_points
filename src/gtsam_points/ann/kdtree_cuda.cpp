// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#include <gtsam_points/ann/kdtree_cuda.hpp>

#include <gtsam_points/cuda/check_error.cuh>
#include <gtsam_points/ann/small_kdtree.hpp>

namespace gtsam_points {

KdTreeGPU::KdTreeGPU(const PointCloud::ConstPtr& points, CUstream_st* stream)
: points(points),
  num_indices(0),
  num_nodes(0),
  indices(nullptr),
  nodes(nullptr) {
  //
  if (!points->has_points()) {
    std::cerr << "error: empty point cloud is given for KdTreeGPU" << std::endl;
    return;
  }
  if (!points->has_points_gpu()) {
    std::cerr << "error: point cloud does not have GPU points for KdTreeGPU" << std::endl;
    return;
  }

  //
  KdTreeBuilder builder;
  UnsafeKdTree<PointCloud> kdtree(*points, builder);

  // copy to GPU
  std::vector<std::uint32_t> h_indices(kdtree.indices.begin(), kdtree.indices.end());
  std::vector<KdTreeNodeGPU> h_nodes(kdtree.nodes.size());

  for (int i = 0; i < kdtree.nodes.size(); i++) {
    const auto& in = kdtree.nodes[i];
    auto& out = h_nodes[i];

    out.left = in.left;
    out.right = in.right;

    if (in.left == INVALID_NODE) {
      out.node_type.lr.first = in.node_type.lr.first;
      out.node_type.lr.last = in.node_type.lr.last;
    } else {
      out.node_type.sub.axis = in.node_type.sub.proj.axis;
      out.node_type.sub.thresh = in.node_type.sub.thresh;
    }
  }

  num_indices = kdtree.indices.size();
  num_nodes = kdtree.nodes.size();
  check_error << cudaMallocAsync(&indices, sizeof(std::uint32_t) * num_indices, stream);
  check_error << cudaMallocAsync(&nodes, sizeof(KdTreeNodeGPU) * num_nodes, stream);
  check_error << cudaMemcpyAsync(indices, h_indices.data(), sizeof(std::uint32_t) * num_indices, cudaMemcpyHostToDevice, stream);
  check_error << cudaMemcpyAsync(nodes, h_nodes.data(), sizeof(KdTreeNodeGPU) * num_nodes, cudaMemcpyHostToDevice, stream);
}

KdTreeGPU::~KdTreeGPU() {
  check_error << cudaFreeAsync(indices, nullptr);
  check_error << cudaFreeAsync(nodes, nullptr);
}

void KdTreeGPU::nearest_neighbor_search_cpu(
  const Eigen::Vector3f* h_queries,
  size_t num_queries,
  std::uint32_t* h_nn_indices,
  float* h_nn_sq_dists,
  CUstream_st* stream) {
  //
  Eigen::Vector3f* d_queries;
  std::uint32_t* d_nn_indices;
  float* d_nn_sq_dists;

  check_error << cudaMallocAsync(&d_queries, sizeof(Eigen::Vector3f) * num_queries, stream);
  check_error << cudaMallocAsync(&d_nn_indices, sizeof(std::uint32_t) * num_queries, stream);
  check_error << cudaMallocAsync(&d_nn_sq_dists, sizeof(float) * num_queries, stream);
  check_error << cudaMemcpyAsync(d_queries, h_queries, sizeof(Eigen::Vector3f) * num_queries, cudaMemcpyHostToDevice, stream);

  nearest_neighbor_search(d_queries, num_queries, d_nn_indices, d_nn_sq_dists, stream);

  check_error << cudaMemcpyAsync(h_nn_indices, d_nn_indices, sizeof(std::uint32_t) * num_queries, cudaMemcpyDeviceToHost, stream);
  check_error << cudaMemcpyAsync(h_nn_sq_dists, d_nn_sq_dists, sizeof(float) * num_queries, cudaMemcpyDeviceToHost, stream);

  check_error << cudaFreeAsync(d_queries, stream);
  check_error << cudaFreeAsync(d_nn_indices, stream);
  check_error << cudaFreeAsync(d_nn_sq_dists, stream);
}

}  // namespace gtsam_points