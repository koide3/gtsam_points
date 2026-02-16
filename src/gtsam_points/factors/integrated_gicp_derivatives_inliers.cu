// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/integrated_gicp_derivatives.cuh>

#include <iostream>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include <gtsam_points/cuda/check_error.cuh>
#include <gtsam_points/cuda/kernels/pose.cuh>
#include <gtsam_points/cuda/stream_temp_buffer_roundrobin.hpp>
#include <gtsam_points/cuda/cuda_malloc_async.hpp>

namespace gtsam_points {

namespace {

struct init_source_target_indices_kernel {
  __host__ __device__ Correspondence operator()(const int i) const { return Correspondence(i, -1); }
};

}  // namespace

void IntegratedGICPDerivatives::reset_inliers(const Eigen::Isometry3f& x, const Eigen::Isometry3f* d_x, bool force_update) {
  touch_points();

  if (
    force_update || source_target_correspondences == nullptr ||
    large_displacement(inlier_evaluation_point, x, inlier_update_thresh_trans, inlier_update_thresh_angle)) {
    inlier_evaluation_point = x;
    inlier_evaluation_point_gpu = d_x;

    check_error << cudaFreeAsync(source_target_correspondences, stream);
    check_error << cudaMallocAsync(&source_target_correspondences, sizeof(Correspondence) * source->size(), stream);
    thrust::transform(
      thrust::cuda::par_nosync.on(stream),
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(source->size()),
      source_target_correspondences,
      init_source_target_indices_kernel());
    num_inliers = source->size();
  } else {
    inlier_evaluation_point_gpu = nullptr;
  }
}

void IntegratedGICPDerivatives::update_inliers(int new_num_inliers) {
  if (inlier_evaluation_point_gpu == nullptr) {
    return;
  }

  // For now, we keep all correspondences without filtering
  // The correspondence filtering can be added later with validity checks
  this->num_inliers = new_num_inliers;
}

}  // namespace gtsam_points