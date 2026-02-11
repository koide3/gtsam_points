// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/integrated_gicp_derivatives.cuh>

#include <iostream>
#include <thrust/remove.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/device/device_reduce.cuh>

#include <gtsam_points/cuda/kernels/pose.cuh>
#include <gtsam_points/cuda/kernels/untie.cuh>
#include <gtsam_points/cuda/kernels/kdtree.cuh>
#include <gtsam_points/cuda/kernels/linearized_system.cuh>
#include <gtsam_points/cuda/kernels/gicp_derivatives.cuh>
#include <gtsam_points/cuda/stream_temp_buffer_roundrobin.hpp>

namespace gtsam_points {

namespace {

/// @brief Kernel to compute source-target correspondence by KdTree NN search
struct kdtree_correspondence_kernel {
  kdtree_correspondence_kernel(
    const Eigen::Isometry3f* linearization_point_ptr,
    const Eigen::Vector3f* source_points,
    const Eigen::Vector3f* target_points,
    const std::uint32_t* target_indices,
    const KdTreeNodeGPU* target_nodes)
  : linearization_point_ptr(linearization_point_ptr),
    source_points(source_points),
    target_points(target_points),
    target_indices(target_indices),
    target_nodes(target_nodes) {}

  __device__ thrust::pair<int, int> operator()(const thrust::pair<int, int>& source_target) const {
    const int source_idx = source_target.first;
    if (source_idx < 0) {
      return source_target;
    }

    const Eigen::Isometry3f& x = *linearization_point_ptr;
    const Eigen::Vector3f transed_pt = x.linear() * source_points[source_idx] + x.translation();

    kdtree_nearest_neighbor_search_kernel nn_search;
    nn_search.points = target_points;
    nn_search.indices = target_indices;
    nn_search.nodes = target_nodes;

    const auto [nn_idx, sq_dist] = nn_search(transed_pt);

    return thrust::make_pair(source_idx, static_cast<int>(nn_idx));
  }

  const Eigen::Isometry3f* linearization_point_ptr;
  const Eigen::Vector3f* source_points;
  const Eigen::Vector3f* target_points;
  const std::uint32_t* target_indices;
  const KdTreeNodeGPU* target_nodes;
};

}  // namespace

void IntegratedGICPDerivatives::issue_compute_error(const Eigen::Isometry3f* d_xl, const Eigen::Isometry3f* d_xe, float* d_output) {
  // Reuse correspondences computed in issue_linearize instead of recomputing KdTree search
  // Compute GICP error using the correspondences
  gicp_error_kernel error_kernel(
    d_xl,
    d_xe,
    reinterpret_cast<const Eigen::Vector3f*>(target->points_gpu),
    reinterpret_cast<const Eigen::Matrix3f*>(target->covs_gpu),
    reinterpret_cast<const Eigen::Vector3f*>(source->points_gpu),
    reinterpret_cast<const Eigen::Matrix3f*>(source->covs_gpu));

  auto first = thrust::make_transform_iterator(source_target_correspondences, error_kernel);

  void* temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, first, d_output, num_inliers, thrust::plus<float>(), 0.0f, stream);

  temp_storage = temp_buffer->get_buffer(temp_storage_bytes);

  cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, first, d_output, num_inliers, thrust::plus<float>(), 0.0f, stream);
}

}  // namespace gtsam_points
