// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_ext/factors/integrated_vgicp_derivatives.cuh>

#include <iostream>
#include <thrust/remove.h>
#include <thrust/host_vector.h>
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

void IntegratedVGICPDerivatives::update_inliers(
  const Eigen::Isometry3f& x,
  const thrust::device_ptr<const Eigen::Isometry3f>& x_ptr,
  bool force_update) {
  if (
    force_update || source_inliears.empty() ||
    large_displacement(inlier_evaluation_point, x, inlier_update_thresh_trans, inlier_update_thresh_angle)) {
    inlier_evaluation_point = x;

    source_inliears.resize(source->size());

    if (enable_surface_validation) {
      lookup_voxels_kernel<true> kernel(
        *target,
        thrust::device_ptr<const Eigen::Vector3f>(source->points_gpu),
        thrust::device_ptr<const Eigen::Vector3f>(source->normals_gpu),
        x_ptr);
      auto corr_first = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), kernel);
      auto corr_last = thrust::make_transform_iterator(thrust::counting_iterator<int>(source->size()), kernel);
      thrust::transform(thrust::cuda::par.on(stream), corr_first, corr_last, source_inliears.begin(), untie_pair_first<int, int>());
    } else {
      lookup_voxels_kernel<false> kernel(
        *target,
        thrust::device_ptr<const Eigen::Vector3f>(source->points_gpu),
        thrust::device_ptr<const Eigen::Vector3f>(source->normals_gpu),
        x_ptr);
      auto corr_first = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), kernel);
      auto corr_last = thrust::make_transform_iterator(thrust::counting_iterator<int>(source->size()), kernel);
      thrust::transform(thrust::cuda::par.on(stream), corr_first, corr_last, source_inliears.begin(), untie_pair_first<int, int>());
    }

    auto remove_loc = thrust::remove(source_inliears.begin(), source_inliears.end(), -1);
    source_inliears.erase(remove_loc, source_inliears.end());
    source_inliears.shrink_to_fit();
  }
}

}