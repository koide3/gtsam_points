#pragma once

#include <Eigen/Core>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <gtsam_ext/types/gaussian_voxelmap.hpp>

namespace gtsam_ext {

struct VoxelMapInfo {
  int num_voxels;
  int num_buckets;
  int max_bucket_scan_count;
  float voxel_resolution;
};

class GaussianVoxelMapGPU : GaussianVoxelMap {
public:
  GaussianVoxelMapGPU(float resolution, int init_num_buckets = 8192 * 2, int max_bucket_scan_count = 10);
  virtual ~GaussianVoxelMapGPU();

  virtual double voxel_resolution() const override { return voxelmap_info.voxel_resolution; }
  virtual void create_voxelmap(const Frame& frame) override;

private:
  void create_bucket_table(cudaStream_t stream, const Frame& frame);

public:
  const int init_num_buckets;
  VoxelMapInfo voxelmap_info;
  thrust::device_vector<VoxelMapInfo> voxelmap_info_ptr;

  thrust::device_vector<thrust::pair<Eigen::Vector3i, int>> buckets;

  // voxel data
  thrust::device_vector<int> num_points;
  thrust::device_vector<Eigen::Vector3f> voxel_means;
  thrust::device_vector<Eigen::Matrix3f> voxel_covs;
};

}