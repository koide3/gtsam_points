// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_ext/factors/integrated_vgicp_derivatives.cuh>

#include <iostream>
#include <thrust/remove.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/device/device_reduce.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include <gtsam_ext/cuda/kernels/pose.cuh>
#include <gtsam_ext/cuda/kernels/untie.cuh>
#include <gtsam_ext/cuda/kernels/lookup_voxels.cuh>
#include <gtsam_ext/cuda/kernels/linearized_system.cuh>
#include <gtsam_ext/cuda/kernels/vgicp_derivatives.cuh>
#include <gtsam_ext/cuda/stream_temp_buffer_roundrobin.hpp>

#include <gtsam_ext/types/gaussian_voxelmap_gpu.hpp>

namespace gtsam_ext {

template <bool enable_surface_validation_>
void IntegratedVGICPDerivatives::issue_compute_error_impl(
  const thrust::device_ptr<const Eigen::Isometry3f>& xl,
  const thrust::device_ptr<const Eigen::Isometry3f>& xe,
  const thrust::device_ptr<float>& output) {
  //
  lookup_voxels_kernel<enable_surface_validation_> corr_kernel(
    *target,
    thrust::device_ptr<const Eigen::Vector3f>(source->points_gpu),
    thrust::device_ptr<const Eigen::Vector3f>(source->normals_gpu),
    xl);
  cub::TransformInputIterator<thrust::pair<int, int>, lookup_voxels_kernel<enable_surface_validation_>, int*> corr_first(
    thrust::raw_pointer_cast(source_inliears.data()),
    corr_kernel);

  vgicp_error_kernel error_kernel(
    xl,
    xe,
    *target,
    thrust::device_ptr<const Eigen::Vector3f>(source->points_gpu),
    thrust::device_ptr<const Eigen::Matrix3f>(source->covs_gpu));
  cub::TransformInputIterator<float, vgicp_error_kernel, decltype(corr_first)> first(corr_first, error_kernel);

  void* temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceReduce::Reduce(
    temp_storage,
    temp_storage_bytes,
    first,
    thrust::raw_pointer_cast(output),
    source_inliears.size(),
    thrust::plus<float>(),
    0.0f,
    stream);

  temp_storage = temp_buffer->get_buffer(temp_storage_bytes);

  cub::DeviceReduce::Reduce(
    temp_storage,
    temp_storage_bytes,
    first,
    thrust::raw_pointer_cast(output),
    source_inliears.size(),
    thrust::plus<float>(),
    0.0f,
    stream);
}

template void IntegratedVGICPDerivatives::issue_compute_error_impl<true>(
  const thrust::device_ptr<const Eigen::Isometry3f>&,
  const thrust::device_ptr<const Eigen::Isometry3f>&,
  const thrust::device_ptr<float>&);
template void IntegratedVGICPDerivatives::issue_compute_error_impl<false>(
  const thrust::device_ptr<const Eigen::Isometry3f>&,
  const thrust::device_ptr<const Eigen::Isometry3f>&,
  const thrust::device_ptr<float>&);

}