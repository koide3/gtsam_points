// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/device/device_reduce.cuh>

#include <gtsam_points/cuda/kernels/lookup_voxels.cuh>
#include <gtsam_points/cuda/kernels/vgicp_derivatives.cuh>
#include <gtsam_points/cuda/stream_temp_buffer_roundrobin.hpp>

namespace gtsam_points {
namespace detail {

template <typename Result, typename Transform>
struct transform_vgicp_linearization {
  transform_vgicp_linearization(vgicp_derivatives_kernel derivatives, Transform transform, const Result& identity)
  : derivatives(derivatives), transform(transform), identity(identity) {}

  __device__ Result operator()(const thrust::pair<int, int>& correspondence) const {
    if (correspondence.first < 0 || correspondence.second < 0) {
      return identity;
    }

    return transform(correspondence, derivatives(correspondence));
  }

  vgicp_derivatives_kernel derivatives;
  Transform transform;
  Result identity;
};

template <typename Result, typename Transform>
struct transform_vgicp_error {
  transform_vgicp_error(vgicp_error_kernel error, Transform transform, const Result& identity)
  : error(error), transform(transform), identity(identity) {}

  __device__ Result operator()(const thrust::pair<int, int>& correspondence) const {
    if (correspondence.first < 0 || correspondence.second < 0) {
      return identity;
    }

    return transform(correspondence, error(correspondence));
  }

  vgicp_error_kernel error;
  Transform transform;
  Result identity;
};

}  // namespace detail

template <typename Result, typename Transform, typename Reduction>
void IntegratedVGICPDerivatives::issue_linearize_transform_reduce(
  const Eigen::Isometry3f* d_x,
  Result* d_output,
  Transform transform,
  Reduction reduction,
  const Result& identity) {
  //
  lookup_voxels_kernel correspondence_lookup(enable_surface_validation, *target, source->points_gpu, source->normals_gpu, d_x);
  auto correspondences = thrust::make_transform_iterator(source_inliers, correspondence_lookup);

  vgicp_derivatives_kernel derivatives(d_x, *target, source->points_gpu, source->covs_gpu);
  detail::transform_vgicp_linearization<Result, Transform> transformed_derivatives(derivatives, transform, identity);
  auto first = thrust::make_transform_iterator(correspondences, transformed_derivatives);

  void* temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, first, d_output, num_inliers, reduction, identity, stream);

  temp_storage = temp_buffer->get_buffer(temp_storage_bytes);
  cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, first, d_output, num_inliers, reduction, identity, stream);
}

template <typename Result, typename Transform>
void IntegratedVGICPDerivatives::issue_linearize_transform_reduce(
  const Eigen::Isometry3f* d_x,
  Result* d_output,
  Transform transform,
  const Result& identity) {
  //
  issue_linearize_transform_reduce(d_x, d_output, transform, thrust::plus<Result>(), identity);
}

template <typename Result, typename Transform, typename Reduction>
void IntegratedVGICPDerivatives::issue_compute_error_transform_reduce(
  const Eigen::Isometry3f* d_xl,
  const Eigen::Isometry3f* d_xe,
  Result* d_output,
  Transform transform,
  Reduction reduction,
  const Result& identity) {
  //
  lookup_voxels_kernel correspondence_lookup(
    enable_surface_validation,
    *target,
    source->points_gpu,
    source->normals_gpu,
    d_xl);
  auto correspondences = thrust::make_transform_iterator(source_inliers, correspondence_lookup);

  vgicp_error_kernel error(d_xl, d_xe, *target, source->points_gpu, source->covs_gpu);
  detail::transform_vgicp_error<Result, Transform> transformed_error(error, transform, identity);
  auto first = thrust::make_transform_iterator(correspondences, transformed_error);

  void* temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, first, d_output, num_inliers, reduction, identity, stream);

  temp_storage = temp_buffer->get_buffer(temp_storage_bytes);
  cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, first, d_output, num_inliers, reduction, identity, stream);
}

template <typename Result, typename Transform>
void IntegratedVGICPDerivatives::issue_compute_error_transform_reduce(
  const Eigen::Isometry3f* d_xl,
  const Eigen::Isometry3f* d_xe,
  Result* d_output,
  Transform transform,
  const Result& identity) {
  //
  issue_compute_error_transform_reduce(d_xl, d_xe, d_output, transform, thrust::plus<Result>(), identity);
}

}  // namespace gtsam_points
