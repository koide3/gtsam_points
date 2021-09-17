#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <thrust/count.h>
#include <thrust/device_new.h>
#include <thrust/device_vector.h>

#include <gtsam_ext/cuda/kernels/vector3_hash.cuh>
#include <gtsam_ext/types/gaussian_voxelmap_gpu.hpp>

namespace gtsam_ext {

struct lookup_voxels_kernel {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  lookup_voxels_kernel(const GaussianVoxelMapGPU& voxelmap, const thrust::device_ptr<const Eigen::Vector3f>& points, const thrust::device_ptr<const Eigen::Isometry3f>& x_ptr)
  : x_ptr(x_ptr),
    voxelmap_info_ptr(voxelmap.voxelmap_info_ptr.data()),
    buckets_ptr(voxelmap.buckets.data()),
    points_ptr(points) {}

  __host__ __device__ thrust::pair<int, int> operator()(int point_idx) const {
    const auto& info = *thrust::raw_pointer_cast(voxelmap_info_ptr);

    const Eigen::Isometry3f& trans = *thrust::raw_pointer_cast(x_ptr);
    const Eigen::Vector3f& x = thrust::raw_pointer_cast(points_ptr)[point_idx];
    const int voxel_idx = lookup_voxel(info.max_bucket_scan_count, info.num_buckets, buckets_ptr, info.voxel_resolution, trans.linear() * x + trans.translation());

    if (voxel_idx < 0) {
      return thrust::make_pair(-1, -1);
    }

    return thrust::make_pair(point_idx, voxel_idx);
  }

  __host__ __device__ void operator()(thrust::pair<int, int>& point_voxel_indices) const {
    const auto& info = *thrust::raw_pointer_cast(voxelmap_info_ptr);

    const int point_idx = thrust::get<0>(point_voxel_indices);
    int& voxel_idx = thrust::get<1>(point_voxel_indices);

    const Eigen::Isometry3f& trans = *thrust::raw_pointer_cast(x_ptr);
    const Eigen::Vector3f& x = thrust::raw_pointer_cast(points_ptr)[point_idx];
    voxel_idx = lookup_voxel(info.max_bucket_scan_count, info.num_buckets, buckets_ptr, info.voxel_resolution, trans.linear() * x + trans.translation());
  }

  thrust::device_ptr<const Eigen::Isometry3f> x_ptr;

  thrust::device_ptr<const VoxelMapInfo> voxelmap_info_ptr;
  thrust::device_ptr<const thrust::pair<Eigen::Vector3i, int>> buckets_ptr;

  thrust::device_ptr<const Eigen::Vector3f> points_ptr;
};

struct invalid_correspondence_kernel {
  __host__ __device__ bool operator()(const thrust::pair<int, int>& point_voxel_indices) const {
    const int point_idx = thrust::get<0>(point_voxel_indices);
    const int voxel_idx = thrust::get<1>(point_voxel_indices);

    return point_idx < 0 || voxel_idx < 0;
  }
};

}  // namespace gtsam_ext
