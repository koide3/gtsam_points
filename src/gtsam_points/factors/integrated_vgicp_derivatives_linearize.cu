// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/integrated_vgicp_derivatives.cuh>

#include <iostream>
#include <thrust/remove.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/device/device_reduce.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include <gtsam_points/cuda/kernels/pose.cuh>
#include <gtsam_points/cuda/kernels/untie.cuh>
#include <gtsam_points/cuda/kernels/lookup_voxels.cuh>
#include <gtsam_points/cuda/kernels/linearized_system.cuh>
#include <gtsam_points/cuda/kernels/vgicp_derivatives.cuh>
#include <gtsam_points/cuda/stream_temp_buffer_roundrobin.hpp>

#include <gtsam_points/types/gaussian_voxelmap_gpu.hpp>

namespace gtsam_points {

template <bool enable_surface_validation_>
void IntegratedVGICPDerivatives::issue_linearize_impl(
  const thrust::device_ptr<const Eigen::Isometry3f>& x,
  const thrust::device_ptr<LinearizedSystem6>& output) {
  //
  lookup_voxels_kernel<enable_surface_validation_> corr_kernel(
    *target,
    thrust::device_ptr<const Eigen::Vector3f>(source->points_gpu),
    thrust::device_ptr<const Eigen::Vector3f>(source->normals_gpu),
    x);
  cub::TransformInputIterator<thrust::pair<int, int>, lookup_voxels_kernel<enable_surface_validation_>, int*> corr_first(source_inliers, corr_kernel);

  vgicp_derivatives_kernel deriv_kernel(
    x,
    *target,
    thrust::device_ptr<const Eigen::Vector3f>(source->points_gpu),
    thrust::device_ptr<const Eigen::Matrix3f>(source->covs_gpu));
  cub::TransformInputIterator<LinearizedSystem6, vgicp_derivatives_kernel, decltype(corr_first)> first(corr_first, deriv_kernel);

  void* temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceReduce::Reduce(
    temp_storage,
    temp_storage_bytes,
    first,
    thrust::raw_pointer_cast(output),
    num_inliers,
    thrust::plus<LinearizedSystem6>(),
    LinearizedSystem6::zero(),
    stream);

  temp_storage = temp_buffer->get_buffer(temp_storage_bytes);

  cub::DeviceReduce::Reduce(
    temp_storage,
    temp_storage_bytes,
    first,
    thrust::raw_pointer_cast(output),
    num_inliers,
    thrust::plus<LinearizedSystem6>(),
    LinearizedSystem6::zero(),
    stream);
}

template void IntegratedVGICPDerivatives::issue_linearize_impl<true>(
  const thrust::device_ptr<const Eigen::Isometry3f>& x,
  const thrust::device_ptr<LinearizedSystem6>& output);
template void IntegratedVGICPDerivatives::issue_linearize_impl<false>(
  const thrust::device_ptr<const Eigen::Isometry3f>& x,
  const thrust::device_ptr<LinearizedSystem6>& output);

}  // namespace gtsam_points