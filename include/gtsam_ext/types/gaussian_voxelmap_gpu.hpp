#pragma once

#include <Eigen/Core>
#include <gtsam_ext/types/gaussian_voxelmap.hpp>

struct CUstream_st;

namespace thrust {

template <typename T1, typename T2>
struct pair;

template <typename T>
class device_allocator;

template <typename T, typename Alloc>
class device_vector;

template <typename T>
struct device_vector_ {
  using type = device_vector<T, device_allocator<T>>;
};

}  // namespace thrust

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
  void create_bucket_table(CUstream_st* stream, const Frame& frame);

public:
  const int init_num_buckets;
  VoxelMapInfo voxelmap_info;
  thrust::device_vector_<VoxelMapInfo> voxelmap_info_ptr;

  thrust::device_vector_<thrust::pair<Eigen::Vector3i, int>> buckets;

  // voxel data
  thrust::device_vector_<int> num_points;
  thrust::device_vector_<Eigen::Vector3f> voxel_means;
  thrust::device_vector_<Eigen::Matrix3f> voxel_covs;
};

}  // namespace gtsam_ext