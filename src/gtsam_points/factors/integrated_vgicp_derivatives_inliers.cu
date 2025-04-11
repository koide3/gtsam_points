// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/integrated_vgicp_derivatives.cuh>

#include <iostream>
#include <thrust/remove.h>
#include <thrust/sequence.h>
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

namespace gtsam_points {

namespace {

struct valid_inlier_kernel {
  valid_inlier_kernel(
    bool enable_surface_validation,
    const GaussianVoxelMapGPU& voxelmap,
    const Eigen::Vector3f* points,
    const Eigen::Vector3f* normals,
    const Eigen::Isometry3f* x_ptr)
  : lookup(enable_surface_validation, voxelmap, points, normals, x_ptr) {}

  __host__ __device__ bool operator()(const int i) const { return lookup(i).first != -1; }

  lookup_voxels_kernel lookup;
};

}  // namespace

void IntegratedVGICPDerivatives::reset_inliers(const Eigen::Isometry3f& x, const Eigen::Isometry3f* d_x, bool force_update) {
  if (
    force_update || source_inliers == nullptr ||
    large_displacement(inlier_evaluation_point, x, inlier_update_thresh_trans, inlier_update_thresh_angle)) {
    inlier_evaluation_point = x;
    inlier_evaluation_point_gpu = d_x;

    check_error << cudaFreeAsync(source_inliers, stream);
    check_error << cudaMallocAsync(&source_inliers, sizeof(int) * source->size(), stream);
    thrust::sequence(thrust::cuda::par_nosync.on(stream), source_inliers, source_inliers + source->size(), 0);
    num_inliers = source->size();
  } else {
    inlier_evaluation_point_gpu = nullptr;
  }
}

void IntegratedVGICPDerivatives::update_inliers(int new_num_inliers) {
  if (inlier_evaluation_point_gpu == nullptr) {
    return;
  }

  int* new_source_inliers;
  check_error << cudaMallocAsync(&new_source_inliers, sizeof(int) * new_num_inliers, stream);

  valid_inlier_kernel kernel(enable_surface_validation, *target, source->points_gpu, source->normals_gpu, inlier_evaluation_point_gpu);

  void* temp_storage = nullptr;
  size_t temp_storage_bytes;

  cub::DeviceSelect::If(
    temp_storage,
    temp_storage_bytes,
    thrust::make_counting_iterator<int>(0),
    new_source_inliers,
    num_inliers_gpu,
    source->size(),
    kernel,
    stream);

  temp_storage = temp_buffer->get_buffer(temp_storage_bytes);

  cub::DeviceSelect::If(
    temp_storage,
    temp_storage_bytes,
    thrust::make_counting_iterator<int>(0),
    new_source_inliers,
    num_inliers_gpu,
    source->size(),
    kernel,
    stream);

  // Validate if the number of inliers is correct. This causes a stream sync.
  // constexpr bool validate = false;
  // if (validate) {
  //   int num_inliers_;
  //   check_error << cudaMemcpyAsync(&num_inliers_, num_inliers_gpu, sizeof(int), cudaMemcpyDeviceToHost, stream);
  //   if (num_inliers_ != new_num_inliers) {
  //     std::cerr << "num_inliers_=" << num_inliers_ << " new_num_inliers=" << new_num_inliers << std::endl;
  //     abort();
  //   }
  // }

  check_error << cudaFreeAsync(source_inliers, stream);
  this->source_inliers = new_source_inliers;
  this->num_inliers = new_num_inliers;
}

}  // namespace gtsam_points