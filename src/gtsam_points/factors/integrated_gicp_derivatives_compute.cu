// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/integrated_gicp_derivatives.cuh>

#include <thrust/iterator/transform_iterator.h>

#include <cub/device/device_reduce.cuh>

#include <gtsam_points/cuda/kernels/gicp_derivatives.cuh>
#include <gtsam_points/cuda/stream_temp_buffer_roundrobin.hpp>

namespace gtsam_points {

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
