// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <Eigen/Core>

#include <gtsam_ext/types/gaussian_voxelmap.hpp>

// forward declaration
struct CUstream_st;

namespace thrust {
template <typename T1, typename T2>
class pair;

template <typename T>
class device_allocator;

template <typename T, typename Alloc>
class device_vector;

}  // namespace thrust

namespace gtsam_ext {

/**
 * @brief Voxel hashmap information
 */
struct VoxelMapInfo {
  int num_voxels;               ///< Number of voxels
  int num_buckets;              ///< Number of buckets
  int max_bucket_scan_count;    ///< Maximum bucket search count
  float voxel_resolution;       ///< Voxel resolution
};

/**
 * @brief Gaussian distribution voxelmap on GPU
 */
class GaussianVoxelMapGPU : public GaussianVoxelMap {
public:
  using Ptr = std::shared_ptr<GaussianVoxelMapGPU>;
  using ConstPtr = std::shared_ptr<const GaussianVoxelMapGPU>;

  /**
   * @param resolution               Voxel resolution
   * @param init_num_buckets         Initial buckets size
   * @param max_bucket_scan_count    Maximum number of open hashmap addressing trials.
   *                                 If a target voxel coord is not found in successive N buckets, it is considered to be not found.
   *                                 Setting a larger value makes the hashing slower but results in a smaller memory footprint.
   * @param target_points_drop_rate  This parameter allows to drop a fraction of points during the bucket creation.
   */
  GaussianVoxelMapGPU(float resolution, int init_num_buckets = 8192 * 2, int max_bucket_scan_count = 10, double target_points_drop_rate = 1e-3);
  virtual ~GaussianVoxelMapGPU();

  /// Voxel resolution
  virtual double voxel_resolution() const override { return voxelmap_info.voxel_resolution; }

  /// Insert a point cloud frame into the voxelmap
  /// @note Incremental insertion is not supported for GPU
  virtual void insert(const Frame& frame) override;

private:
  void create_bucket_table(CUstream_st* stream, const Frame& frame);

public:
  using Indices = thrust::device_vector<int, thrust::device_allocator<int>>;
  using Points = thrust::device_vector<Eigen::Vector3f, thrust::device_allocator<Eigen::Vector3f>>;
  using Matrices = thrust::device_vector<Eigen::Matrix3f, thrust::device_allocator<Eigen::Matrix3f>>;
  using Buckets = thrust::device_vector<thrust::pair<Eigen::Vector3i, int>, thrust::device_allocator<thrust::pair<Eigen::Vector3i, int>>>;
  using VoxelMapInfoVec = thrust::device_vector<VoxelMapInfo, thrust::device_allocator<VoxelMapInfo>>;

  const int init_num_buckets;                           ///< Initial number of buckets
  const double target_points_drop_rate;                 ///< Allowable points drop rate
  VoxelMapInfo voxelmap_info;                           ///< Voxelmap information
  std::unique_ptr<VoxelMapInfoVec> voxelmap_info_ptr;   ///< Voxelmap information on GPU memory

  std::unique_ptr<Buckets> buckets;                     ///< Voxel buckets for hashing

  // voxel data
  std::unique_ptr<Indices> num_points;      ///< Number of points in eac voxel
  std::unique_ptr<Points> voxel_means;      ///< Voxel means
  std::unique_ptr<Points> voxel_normals;    ///< Voxel normals
  std::unique_ptr<Matrices> voxel_covs;     ///< Voxel covariances
};

}