// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <Eigen/Core>

#include <gtsam_points/types/gaussian_voxelmap.hpp>

// forward declaration
struct CUstream_st;

namespace gtsam_points {

/**
 * @brief Voxel hashmap information
 */
struct VoxelMapInfo {
  int num_voxels;             ///< Number of voxels
  int num_buckets;            ///< Number of buckets
  int max_bucket_scan_count;  ///< Maximum bucket search count
  float voxel_resolution;     ///< Voxel resolution
};

/**
 * @brief Voxel bucket (avoid using thrust::pair for CUDA compatibility)
 */
struct VoxelBucket {
  Eigen::Vector3i first;
  int second;
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
  GaussianVoxelMapGPU(
    float resolution,
    int init_num_buckets = 8192 * 2,
    int max_bucket_scan_count = 10,
    double target_points_drop_rate = 1e-3,
    CUstream_st* stream = 0);
  virtual ~GaussianVoxelMapGPU();

  /// Voxel resolution
  virtual double voxel_resolution() const override { return voxelmap_info.voxel_resolution; }

  /// Insert a point cloud frame into the voxelmap
  /// @note Incremental insertion is not supported for GPU
  virtual void insert(const PointCloud& frame) override;

private:
  void create_bucket_table(CUstream_st* stream, const PointCloud& frame);

public:
  CUstream_st* stream;

  const int init_num_buckets;                   ///< Initial number of buckets
  const double target_points_drop_rate;         ///< Allowable points drop rate
  VoxelMapInfo voxelmap_info;                   ///< Voxelmap information
  VoxelMapInfo* voxelmap_info_ptr;              ///< Voxelmap information on GPU memory

  VoxelBucket* buckets;                         ///< Voxel buckets for hashing

  // voxel data
  int* num_points;               ///< Number of points in eac voxel
  Eigen::Vector3f* voxel_means;  ///< Voxel means
  Eigen::Matrix3f* voxel_covs;   ///< Voxel covariances
};

std::vector<Eigen::Vector3f> download_voxel_means(const GaussianVoxelMapGPU& voxelmap, CUstream_st* stream = nullptr);
std::vector<Eigen::Matrix3f> download_voxel_covs(const GaussianVoxelMapGPU& voxelmap, CUstream_st* stream = nullptr);

}  // namespace gtsam_points