// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/integrated_vgicp_derivatives.cuh>

#include <iostream>
#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include <gtsam_points/cuda/check_error.cuh>
#include <gtsam_points/cuda/kernels/pose.cuh>
#include <gtsam_points/cuda/kernels/untie.cuh>
#include <gtsam_points/cuda/kernels/lookup_voxels.cuh>
#include <gtsam_points/cuda/kernels/linearized_system.cuh>
#include <gtsam_points/cuda/kernels/vgicp_derivatives.cuh>
#include <gtsam_points/cuda/stream_temp_buffer_roundrobin.hpp>
#include <gtsam_points/cuda/cuda_malloc_async.hpp>

#include <gtsam_points/types/gaussian_voxelmap_gpu.hpp>

struct valid_inlier_kernel {
  __host__ __device__ bool operator()(const int i) const { return i != -1; }
};

namespace gtsam_points {

void IntegratedVGICPDerivatives::update_inliers(const Eigen::Isometry3f& x, const Eigen::Isometry3f* d_x, bool force_update) {
  if (
    force_update || source_inliers == nullptr ||
    large_displacement(inlier_evaluation_point, x, inlier_update_thresh_trans, inlier_update_thresh_angle)) {
    inlier_evaluation_point = x;

    int* inliers;
    int* valid_inliers;
    check_error << cudaMallocAsync(&inliers, sizeof(int) * source->size(), stream);
    check_error << cudaMallocAsync(&valid_inliers, sizeof(int) * source->size(), stream);

    if (enable_surface_validation) {
      lookup_voxels_kernel<true> kernel(*target, source->points_gpu, source->normals_gpu, d_x);
      auto corr_first = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), kernel);
      auto corr_last = thrust::make_transform_iterator(thrust::counting_iterator<int>(source->size()), kernel);
      thrust::transform(thrust::cuda::par_nosync.on(stream), corr_first, corr_last, inliers, untie_pair_first<int, int>());
    } else {
      lookup_voxels_kernel<false> kernel(*target, source->points_gpu, source->normals_gpu, d_x);
      auto corr_first = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), kernel);
      auto corr_last = thrust::make_transform_iterator(thrust::counting_iterator<int>(source->size()), kernel);
      thrust::transform(thrust::cuda::par_nosync.on(stream), corr_first, corr_last, inliers, untie_pair_first<int, int>());
    }

    void* temp_storage = nullptr;
    size_t temp_storage_bytes;

    cub::DeviceSelect::If(temp_storage, temp_storage_bytes, inliers, valid_inliers, num_inliers_gpu, source->size(), valid_inlier_kernel(), stream);
    temp_storage = temp_buffer->get_buffer(temp_storage_bytes);
    cub::DeviceSelect::If(temp_storage, temp_storage_bytes, inliers, valid_inliers, num_inliers_gpu, source->size(), valid_inlier_kernel(), stream);

    check_error << cudaMemcpyAsync(&num_inliers, num_inliers_gpu, sizeof(int), cudaMemcpyDeviceToHost, stream);
    check_error << cudaStreamSynchronize(stream);

    check_error << cudaFreeAsync(source_inliers, stream);
    check_error << cudaMallocAsync(&source_inliers, sizeof(int) * num_inliers, stream);
    check_error << cudaMemcpyAsync(source_inliers, valid_inliers, sizeof(int) * num_inliers, cudaMemcpyDeviceToDevice, stream);
    check_error << cudaFreeAsync(inliers, stream);
    check_error << cudaFreeAsync(valid_inliers, stream);
  }
}

}  // namespace gtsam_points