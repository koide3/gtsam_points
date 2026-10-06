// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/integrated_vgicp_derivatives.cuh>

#include <gtsam_points/cuda/kernels/linearized_system.cuh>

namespace gtsam_points {

namespace {

struct identity_linearization_transform {
  __device__ LinearizedSystem6 operator()(
    const thrust::pair<int, int>&,
    const LinearizedSystem6& linearized) const {
    //
    return linearized;
  }
};

}  // namespace

void IntegratedVGICPDerivatives::issue_linearize(const Eigen::Isometry3f* d_x, LinearizedSystem6* d_output) {
  issue_linearize_transform_reduce(d_x, d_output, identity_linearization_transform(), LinearizedSystem6::zero());
}

}  // namespace gtsam_points
