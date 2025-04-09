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

void IntegratedVGICPDerivatives::issue_compute_error(const Eigen::Isometry3f* d_xl, const Eigen::Isometry3f* d_xe, float* d_output) {
  //
  lookup_voxels_kernel corr_kernel(enable_surface_validation, *target, source->points_gpu, source->normals_gpu, d_xl);
  cub::TransformInputIterator<thrust::pair<int, int>, lookup_voxels_kernel, int*> corr_first(source_inliers, corr_kernel);

  vgicp_error_kernel error_kernel(d_xl, d_xe, *target, source->points_gpu, source->covs_gpu);
  cub::TransformInputIterator<float, vgicp_error_kernel, decltype(corr_first)> first(corr_first, error_kernel);

  void* temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, first, d_output, num_inliers, thrust::plus<float>(), 0.0f, stream);

  temp_storage = temp_buffer->get_buffer(temp_storage_bytes);

  cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, first, d_output, num_inliers, thrust::plus<float>(), 0.0f, stream);
}

}  // namespace gtsam_points