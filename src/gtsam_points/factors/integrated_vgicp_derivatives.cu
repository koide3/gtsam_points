// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/integrated_vgicp_derivatives.cuh>

#include <iostream>

#include <gtsam_points/cuda/check_error.cuh>
#include <gtsam_points/cuda/kernels/linearized_system.cuh>
#include <gtsam_points/cuda/kernels/vgicp_derivatives.cuh>
#include <gtsam_points/cuda/stream_temp_buffer_roundrobin.hpp>
#include <gtsam_points/cuda/cuda_malloc_async.hpp>

#include <gtsam_points/types/gaussian_voxelmap_gpu.hpp>

namespace gtsam_points {

IntegratedVGICPDerivatives::IntegratedVGICPDerivatives(
  const GaussianVoxelMapGPU::ConstPtr& target,
  const PointCloud::ConstPtr& source,
  CUstream_st* ext_stream,
  std::shared_ptr<TempBufferManager> temp_buffer)
: enable_surface_validation(false),
  inlier_update_thresh_trans(1e-6),
  inlier_update_thresh_angle(1e-6),
  target(target),
  source(source),
  external_stream(true),
  stream(ext_stream),
  temp_buffer(temp_buffer),
  num_inliers(0),
  source_inliers(nullptr) {
  //
  if (stream == nullptr) {
    external_stream = false;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }

  if (this->temp_buffer == nullptr) {
    this->temp_buffer.reset(new TempBufferManager());
  }

  check_error << cudaMallocAsync(&num_inliers_gpu, sizeof(int), stream);
  // check_error << cudaHostRegister(&num_inliers, sizeof(int), cudaHostRegisterDefault);
}

IntegratedVGICPDerivatives::~IntegratedVGICPDerivatives() {
  check_error << cudaFreeAsync(source_inliers, stream);
  check_error << cudaFreeAsync(num_inliers_gpu, stream);
  // check_error << cudaHostUnregister(&num_inliers);

  if (!external_stream) {
    cudaStreamDestroy(stream);
  }
}

void IntegratedVGICPDerivatives::sync_stream() {
  check_error << cudaStreamSynchronize(stream);
}

LinearizedSystem6 IntegratedVGICPDerivatives::linearize(const Eigen::Isometry3f& x) {
  thrust::device_vector<Eigen::Isometry3f> x_ptr(1);
  thrust::device_vector<LinearizedSystem6> output_ptr(1);

  x_ptr[0] = x;

  reset_inliers(x, thrust::raw_pointer_cast(x_ptr.data()));
  issue_linearize(thrust::raw_pointer_cast(x_ptr.data()), thrust::raw_pointer_cast(output_ptr.data()));
  sync_stream();

  LinearizedSystem6 linearized = output_ptr[0];

  return linearized;
}

double IntegratedVGICPDerivatives::compute_error(const Eigen::Isometry3f& d_xl, const Eigen::Isometry3f& d_xe) {
  thrust::device_vector<Eigen::Isometry3f> xs_ptr(2);
  xs_ptr[0] = d_xl;
  xs_ptr[1] = d_xe;
  thrust::device_vector<float> output_ptr(1);

  issue_compute_error(
    thrust::raw_pointer_cast(xs_ptr.data()),
    thrust::raw_pointer_cast(xs_ptr.data() + 1),
    thrust::raw_pointer_cast(output_ptr.data()));
  sync_stream();

  float error = output_ptr[0];
  return error;
}

}  // namespace gtsam_points