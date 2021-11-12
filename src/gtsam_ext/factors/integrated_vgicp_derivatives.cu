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

IntegratedVGICPDerivatives::IntegratedVGICPDerivatives(
  const VoxelizedFrame::ConstPtr& target,
  const Frame::ConstPtr& source,
  CUstream_st* ext_stream,
  std::shared_ptr<TempBufferManager> temp_buffer)
: inlier_update_thresh_trans(1e-3),
  inlier_update_thresh_angle(1e-3),
  target(target),
  source(source),
  external_stream(true),
  stream(ext_stream),
  temp_buffer(temp_buffer) {
  //
  if (stream == nullptr) {
    external_stream = false;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }

  if (this->temp_buffer == nullptr) {
    this->temp_buffer.reset(new TempBufferManager());
  }
}

IntegratedVGICPDerivatives::~IntegratedVGICPDerivatives() {
  if (!external_stream) {
    cudaStreamDestroy(stream);
  }
}

void IntegratedVGICPDerivatives::sync_stream() {
  cudaStreamSynchronize(stream);
}

LinearizedSystem6 IntegratedVGICPDerivatives::linearize(const Eigen::Isometry3f& x) {
  thrust::device_vector<Eigen::Isometry3f> x_ptr(1);
  thrust::device_vector<LinearizedSystem6> output_ptr(1);

  x_ptr[0] = x;

  update_inliers(x, x_ptr.data());
  issue_linearize(x_ptr.data(), output_ptr.data());
  sync_stream();

  LinearizedSystem6 linearized = output_ptr[0];

  return linearized;
}

double IntegratedVGICPDerivatives::compute_error(const Eigen::Isometry3f& xl, const Eigen::Isometry3f& xe) {
  thrust::device_vector<Eigen::Isometry3f> xs_ptr(2);
  xs_ptr[0] = xl;
  xs_ptr[1] = xe;
  thrust::device_vector<float> output_ptr(1);

  issue_compute_error(xs_ptr.data(), xs_ptr.data() + 1, output_ptr.data());
  sync_stream();

  float error = output_ptr[0];
  return error;
}

void IntegratedVGICPDerivatives::update_inliers(const Eigen::Isometry3f& x, const thrust::device_ptr<const Eigen::Isometry3f>& x_ptr, bool force_update) {
  if (force_update || source_inliears.empty() || large_displacement(inlier_evaluation_point, x, inlier_update_thresh_trans, inlier_update_thresh_angle)) {
    inlier_evaluation_point = x;

    source_inliears.resize(source->size());

    lookup_voxels_kernel kernel(*target->voxels_gpu, thrust::device_ptr<const Eigen::Vector3f>(source->points_gpu), x_ptr);
    auto corr_first = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), kernel);
    auto corr_last = thrust::make_transform_iterator(thrust::counting_iterator<int>(source->size()), kernel);

    thrust::transform(thrust::cuda::par.on(stream), corr_first, corr_last, source_inliears.begin(), untie_pair_first<int, int>());
    auto remove_loc = thrust::remove(source_inliears.begin(), source_inliears.end(), -1);

    source_inliears.erase(remove_loc, source_inliears.end());
    source_inliears.shrink_to_fit();
  }
}

void IntegratedVGICPDerivatives::issue_linearize(const thrust::device_ptr<const Eigen::Isometry3f>& x, const thrust::device_ptr<LinearizedSystem6>& output) {
  lookup_voxels_kernel corr_kernel(*target->voxels_gpu, thrust::device_ptr<const Eigen::Vector3f>(source->points_gpu), x);
  cub::TransformInputIterator<thrust::pair<int, int>, lookup_voxels_kernel, int*> corr_first(thrust::raw_pointer_cast(source_inliears.data()), corr_kernel);

  vgicp_derivatives_kernel deriv_kernel(
    x,
    *target->voxels_gpu,
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
    source_inliears.size(),
    thrust::plus<LinearizedSystem6>(),
    LinearizedSystem6::zero(),
    stream);

  temp_storage = temp_buffer->get_buffer(temp_storage_bytes);

  cub::DeviceReduce::Reduce(
    temp_storage,
    temp_storage_bytes,
    first,
    thrust::raw_pointer_cast(output),
    source_inliears.size(),
    thrust::plus<LinearizedSystem6>(),
    LinearizedSystem6::zero(),
    stream);
}

void IntegratedVGICPDerivatives::issue_compute_error(
  const thrust::device_ptr<const Eigen::Isometry3f>& xl,
  const thrust::device_ptr<const Eigen::Isometry3f>& xe,
  const thrust::device_ptr<float>& output) {
  //
  lookup_voxels_kernel corr_kernel(*target->voxels_gpu, thrust::device_ptr<const Eigen::Vector3f>(source->points_gpu), xl);
  cub::TransformInputIterator<thrust::pair<int, int>, lookup_voxels_kernel, int*> corr_first(thrust::raw_pointer_cast(source_inliears.data()), corr_kernel);

  vgicp_error_kernel error_kernel(
    xl,
    xe,
    *target->voxels_gpu,
    thrust::device_ptr<const Eigen::Vector3f>(source->points_gpu),
    thrust::device_ptr<const Eigen::Matrix3f>(source->covs_gpu));
  cub::TransformInputIterator<float, vgicp_error_kernel, decltype(corr_first)> first(corr_first, error_kernel);

  void* temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, first, thrust::raw_pointer_cast(output), source_inliears.size(), thrust::plus<float>(), 0.0f, stream);

  temp_storage = temp_buffer->get_buffer(temp_storage_bytes);

  cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, first, thrust::raw_pointer_cast(output), source_inliears.size(), thrust::plus<float>(), 0.0f, stream);
}

}  // namespace gtsam_ext