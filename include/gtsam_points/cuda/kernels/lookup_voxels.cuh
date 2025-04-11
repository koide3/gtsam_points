// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <thrust/pair.h>
#include <thrust/count.h>
#include <thrust/device_new.h>
#include <thrust/device_vector.h>

#include <gtsam_points/cuda/kernels/vector3_hash.cuh>
#include <gtsam_points/types/gaussian_voxelmap_gpu.hpp>

namespace gtsam_points {

struct lookup_voxels_kernel {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  lookup_voxels_kernel(
    bool enable_surface_validation,
    const GaussianVoxelMapGPU& voxelmap,
    const Eigen::Vector3f* points,
    const Eigen::Vector3f* normals,
    const Eigen::Isometry3f* x_ptr)
  : enable_surface_validation(enable_surface_validation),
    x_ptr(x_ptr),
    voxelmap_info_ptr(voxelmap.voxelmap_info_ptr),
    buckets_ptr(voxelmap.buckets),
    points_ptr(points),
    normals_ptr(normals) {}

  __host__ __device__ thrust::pair<int, int> operator()(int point_idx) const {
    const auto& info = *voxelmap_info_ptr;

    const Eigen::Isometry3f& trans = *x_ptr;
    const Eigen::Vector3f& x = points_ptr[point_idx];
    const Eigen::Vector3f transed_x = trans.linear() * x + trans.translation();

    if (enable_surface_validation) {
      const Eigen::Vector3f& normal = normals_ptr[point_idx];
      const Eigen::Vector3f transed_normal = trans.linear() * normal;

      // 0.17 = cos(10 deg)
      if (transed_x.normalized().dot(transed_normal) > surface_validation_thresh) {
        return thrust::make_pair(-1, -1);
      }
    }

    const int voxel_idx = lookup_voxel(info.max_bucket_scan_count, info.num_buckets, buckets_ptr, info.voxel_resolution, transed_x);
    if (voxel_idx < 0) {
      return thrust::make_pair(-1, -1);
    }

    return thrust::make_pair(point_idx, voxel_idx);
  }

  __host__ __device__ void operator()(thrust::pair<int, int>& point_voxel_indices) const {
    const auto& info = *voxelmap_info_ptr;

    const int point_idx = thrust::get<0>(point_voxel_indices);
    int& voxel_idx = thrust::get<1>(point_voxel_indices);

    const Eigen::Isometry3f& trans = *x_ptr;
    const Eigen::Vector3f& x = points_ptr[point_idx];
    const Eigen::Vector3f transed_x = trans.linear() * x + trans.translation();

    if (enable_surface_validation) {
      const Eigen::Vector3f& normal = normals_ptr[point_idx];
      const Eigen::Vector3f transed_normal = trans.linear() * normal;

      if (transed_x.normalized().dot(transed_normal) > surface_validation_thresh) {
        voxel_idx = -1;
        return;
      }
    }

    voxel_idx = lookup_voxel(info.max_bucket_scan_count, info.num_buckets, buckets_ptr, info.voxel_resolution, transed_x);
  }

  // f(x) = cos(x * M_PI / 180.0)
  // f(90) = 0.000   f(80) = 0.174   f(70) = 0.343   f(60) = 0.500
  // f(50) = 0.643   f(40) = 0.766   f(30) = 0.866   f(20) = 0.940
  static const float constexpr surface_validation_thresh = 0.174;  // cos(80.0 * M_PI / 180.0)

  const bool enable_surface_validation;

  const Eigen::Isometry3f* x_ptr;

  const VoxelMapInfo* voxelmap_info_ptr;
  const VoxelBucket* buckets_ptr;

  const Eigen::Vector3f* points_ptr;
  const Eigen::Vector3f* normals_ptr;
};

struct invalid_correspondence_kernel {
  __host__ __device__ bool operator()(const thrust::pair<int, int>& point_voxel_indices) const {
    const int point_idx = thrust::get<0>(point_voxel_indices);
    const int voxel_idx = thrust::get<1>(point_voxel_indices);

    return point_idx < 0 || voxel_idx < 0;
  }
};

}  // namespace gtsam_points
