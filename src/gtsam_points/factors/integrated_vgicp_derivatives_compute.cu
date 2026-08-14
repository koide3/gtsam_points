// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/integrated_vgicp_derivatives.cuh>

#include <thrust/pair.h>

namespace gtsam_points {

namespace {

struct identity_error_transform {
  __device__ float operator()(const thrust::pair<int, int>&, float error) const { return error; }
};

}  // namespace

void IntegratedVGICPDerivatives::issue_compute_error(const Eigen::Isometry3f* d_xl, const Eigen::Isometry3f* d_xe, float* d_output) {
  issue_compute_error_transform_reduce(d_xl, d_xe, d_output, identity_error_transform(), 0.0f);
}

}  // namespace gtsam_points
