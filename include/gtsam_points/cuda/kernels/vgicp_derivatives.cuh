// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <Eigen/Core>
#include <thrust/device_vector.h>

#include <gtsam_points/cuda/kernels/pose.cuh>
#include <gtsam_points/cuda/kernels/linearized_system.cuh>
#include <gtsam_points/types/gaussian_voxelmap_gpu.hpp>

namespace gtsam_points {

struct vgicp_derivatives_kernel {
  vgicp_derivatives_kernel(
    const Eigen::Isometry3f* linearization_point_ptr,
    const GaussianVoxelMapGPU& voxelmap,
    const Eigen::Vector3f* source_means,
    const Eigen::Matrix3f* source_covs)
  : linearization_point_ptr(linearization_point_ptr),
    voxel_num_points_ptr(voxelmap.num_points),
    voxel_means_ptr(voxelmap.voxel_means),
    voxel_covs_ptr(voxelmap.voxel_covs),
    source_means_ptr(source_means),
    source_covs_ptr(source_covs) {}

  __device__ LinearizedSystem6 operator()(const thrust::pair<int, int>& source_target_correspondence) const {
    const int source_idx = source_target_correspondence.first;
    const int target_idx = source_target_correspondence.second;
    if (source_idx < 0 || target_idx < 0) {
      return LinearizedSystem6::zero();
    }

    const Eigen::Isometry3f& x = *linearization_point_ptr;
    const Eigen::Matrix3f R = x.linear();
    const Eigen::Vector3f t = x.translation();

    const Eigen::Vector3f& mean_A = source_means_ptr[source_idx];
    const Eigen::Matrix3f& cov_A = source_covs_ptr[source_idx];
    const Eigen::Vector3f transed_mean_A = R * mean_A + t;

    const Eigen::Vector3f& mean_B = voxel_means_ptr[target_idx];
    const Eigen::Matrix3f& cov_B = voxel_covs_ptr[target_idx];

    const int num_points = voxel_num_points_ptr[target_idx];

    const Eigen::Matrix3f RCR = (R * cov_A * R.transpose());
    const Eigen::Matrix3f RCR_inv = (cov_B + RCR).inverse();
    Eigen::Vector3f error = mean_B - transed_mean_A;

    Eigen::Matrix<float, 3, 6> J_target;
    J_target.block<3, 3>(0, 0) = -skew_symmetric(transed_mean_A);
    J_target.block<3, 3>(0, 3) = Eigen::Matrix3f::Identity();

    Eigen::Matrix<float, 3, 6> J_source;
    J_source.block<3, 3>(0, 0) = R * skew_symmetric(mean_A);
    J_source.block<3, 3>(0, 3) = -R;

    Eigen::Matrix<float, 6, 3> J_target_RCR_inv = J_target.transpose() * RCR_inv;
    Eigen::Matrix<float, 6, 3> J_source_RCR_inv = J_source.transpose() * RCR_inv;

    LinearizedSystem6 linearized;
    linearized.num_inliers = 1;
    linearized.error = error.transpose() * RCR_inv * error;
    linearized.H_target = J_target_RCR_inv * J_target;
    linearized.H_source = J_source_RCR_inv * J_source;
    linearized.H_target_source = J_target_RCR_inv * J_source;
    linearized.b_target = J_target_RCR_inv * error;
    linearized.b_source = J_source_RCR_inv * error;

    return linearized;
  }

  const Eigen::Isometry3f* linearization_point_ptr;

  const int* voxel_num_points_ptr;
  const Eigen::Vector3f* voxel_means_ptr;
  const Eigen::Matrix3f* voxel_covs_ptr;

  const Eigen::Vector3f* source_means_ptr;
  const Eigen::Matrix3f* source_covs_ptr;
};

struct vgicp_error_kernel {
  vgicp_error_kernel(
    const Eigen::Isometry3f* linearization_point_ptr,
    const Eigen::Isometry3f* evaluation_point_ptr,
    const GaussianVoxelMapGPU& voxelmap,
    const Eigen::Vector3f* source_means,
    const Eigen::Matrix3f* source_covs)
  : linearization_point_ptr(linearization_point_ptr),
    evaluation_point_ptr(evaluation_point_ptr),
    voxel_num_points_ptr(voxelmap.num_points),
    voxel_means_ptr(voxelmap.voxel_means),
    voxel_covs_ptr(voxelmap.voxel_covs),
    source_means_ptr(source_means),
    source_covs_ptr(source_covs) {}

  __device__ float operator()(const thrust::pair<int, int>& source_target_correspondence) const {
    const int source_idx = source_target_correspondence.first;
    const int target_idx = source_target_correspondence.second;
    if (source_idx < 0 || target_idx < 0) {
      return 0.0f;
    }

    const Eigen::Isometry3f& xl = *linearization_point_ptr;
    const Eigen::Matrix3f Rl = xl.linear();

    const Eigen::Isometry3f& xe = *evaluation_point_ptr;
    const Eigen::Matrix3f Re = xe.linear();
    const Eigen::Vector3f te = xe.translation();

    const Eigen::Vector3f& mean_A = source_means_ptr[source_idx];
    const Eigen::Matrix3f& cov_A = source_covs_ptr[source_idx];
    const Eigen::Vector3f transed_mean_A = Re * mean_A + te;

    const Eigen::Vector3f& mean_B = voxel_means_ptr[target_idx];
    const Eigen::Matrix3f& cov_B = voxel_covs_ptr[target_idx];

    const int num_points = voxel_num_points_ptr[target_idx];

    const Eigen::Matrix3f RCR = (Rl * cov_A * Rl.transpose());
    const Eigen::Matrix3f RCR_inv = (cov_B + RCR).inverse();
    Eigen::Vector3f error = mean_B - transed_mean_A;

    return error.transpose() * RCR_inv * error;
  }

  const Eigen::Isometry3f* linearization_point_ptr;
  const Eigen::Isometry3f* evaluation_point_ptr;

  const int* voxel_num_points_ptr;
  const Eigen::Vector3f* voxel_means_ptr;
  const Eigen::Matrix3f* voxel_covs_ptr;

  const Eigen::Vector3f* source_means_ptr;
  const Eigen::Matrix3f* source_covs_ptr;
};

}  // namespace gtsam_points
