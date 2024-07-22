// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/types/gaussian_voxelmap_gpu.hpp>

#include <thrust/pair.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <set>
#include <chrono>
#include <device_atomic_functions.h>
#include <gtsam_points/cuda/check_error.cuh>
#include <gtsam_points/cuda/kernels/vector3_hash.cuh>
#include <gtsam_points/cuda/cuda_malloc_async.hpp>

namespace gtsam_points {

// point coord -> voxel coord conversion
struct voxel_coord_kernel {
  voxel_coord_kernel(const VoxelMapInfo* info) : voxelmap_info_ptr(info) {}

  __host__ __device__ Eigen::Vector3i operator()(const Eigen::Vector3f& x) const {
    const auto& info = *thrust::raw_pointer_cast(voxelmap_info_ptr);
    return calc_voxel_coord(x, info.voxel_resolution);
  }

  const thrust::device_ptr<const VoxelMapInfo> voxelmap_info_ptr;
};

// assign voxel indices to buckets
struct voxel_bucket_assignment_kernel {
  voxel_bucket_assignment_kernel(
    const VoxelMapInfo* voxelmap_info,
    const Eigen::Vector3i* point_coords,
    thrust::pair<int, int>* index_buckets,
    int* voxels_failures)
  : voxelmap_info_ptr(voxelmap_info),
    point_coords_ptr(point_coords),
    index_buckets_ptr(index_buckets),
    voxels_failures_ptr(voxels_failures) {}

  __device__ void operator()(int point_index) const {
    const auto& info = *thrust::raw_pointer_cast(voxelmap_info_ptr);
    const Eigen::Vector3i* coords = thrust::raw_pointer_cast(point_coords_ptr);
    uint64_t hash = vector3i_hash(coords[point_index]);

    for (int i = 0; i < info.max_bucket_scan_count; i++) {
      uint64_t bucket_index = (hash + i) % info.num_buckets;
      thrust::pair<int, int>* index_bucket = thrust::raw_pointer_cast(index_buckets_ptr) + bucket_index;

      int old = atomicCAS(&index_bucket->first, -1, point_index);
      if (old < 0) {
        index_bucket->second = atomicAdd(thrust::raw_pointer_cast(voxels_failures_ptr), 1);
        return;
      }

      if (equal(coords[point_index], coords[old])) {
        return;
      }
    }
    atomicAdd(thrust::raw_pointer_cast(voxels_failures_ptr) + 1, 1);
  }

  thrust::device_ptr<const VoxelMapInfo> voxelmap_info_ptr;
  thrust::device_ptr<const Eigen::Vector3i> point_coords_ptr;
  thrust::device_ptr<thrust::pair<int, int>> index_buckets_ptr;
  thrust::device_ptr<int> voxels_failures_ptr;
};

// pair<point index, bucket index>  to pair<voxel coord, bucket index>
struct voxel_coord_select_kernel {
  voxel_coord_select_kernel(const Eigen::Vector3i* point_coords) : point_coords_ptr(point_coords) {}

  __device__ VoxelBucket operator()(const thrust::pair<int, int>& index_bucket) const {
    if (index_bucket.first < 0) {
      return {Eigen::Vector3i(0, 0, 0), -1};
    }

    return {thrust::raw_pointer_cast(point_coords_ptr)[index_bucket.first], index_bucket.second};
  }

  thrust::device_ptr<const Eigen::Vector3i> point_coords_ptr;
};

// accumulate points and covs
struct accumulate_points_kernel {
  accumulate_points_kernel(
    const VoxelMapInfo* voxelmap_info_ptr,
    const VoxelBucket* buckets,
    int* num_points,
    Eigen::Vector3f* voxel_means,
    Eigen::Matrix3f* voxel_covs)
  : voxelmap_info_ptr(voxelmap_info_ptr),
    buckets_ptr(buckets),
    num_points_ptr(num_points),
    voxel_means_ptr(voxel_means),
    voxel_covs_ptr(voxel_covs) {}

  __device__ void operator()(const thrust::tuple<Eigen::Vector3f, Eigen::Matrix3f>& input) const {
    const auto& info = *thrust::raw_pointer_cast(voxelmap_info_ptr);

    const auto& mean = thrust::get<0>(input);
    const auto& cov = thrust::get<1>(input);

    const Eigen::Vector3i coord = calc_voxel_coord(mean, info.voxel_resolution);
    uint64_t hash = vector3i_hash(coord);

    for (int i = 0; i < info.max_bucket_scan_count; i++) {
      uint64_t bucket_index = (hash + i) % info.num_buckets;
      const VoxelBucket& bucket = thrust::raw_pointer_cast(buckets_ptr)[bucket_index];

      if (bucket.second < 0) {
        break;
      }

      if (equal(bucket.first, coord)) {
        int& num_points = thrust::raw_pointer_cast(num_points_ptr)[bucket.second];
        Eigen::Vector3f& voxel_mean = thrust::raw_pointer_cast(voxel_means_ptr)[bucket.second];
        Eigen::Matrix3f& voxel_cov = thrust::raw_pointer_cast(voxel_covs_ptr)[bucket.second];

        atomicAdd(&num_points, 1);
        for (int j = 0; j < 3; j++) {
          atomicAdd(voxel_mean.data() + j, mean[j]);
        }

        for (int j = 0; j < 9; j++) {
          atomicAdd(voxel_cov.data() + j, cov.data()[j]);
        }
      }
    }
  }

  thrust::device_ptr<const VoxelMapInfo> voxelmap_info_ptr;
  thrust::device_ptr<const VoxelBucket> buckets_ptr;

  thrust::device_ptr<int> num_points_ptr;
  thrust::device_ptr<Eigen::Vector3f> voxel_means_ptr;
  thrust::device_ptr<Eigen::Matrix3f> voxel_covs_ptr;
};

struct finalize_voxels_kernel {
  finalize_voxels_kernel(int* num_points, Eigen::Vector3f* voxel_means, Eigen::Matrix3f* voxel_covs)
  : num_points_ptr(num_points),
    voxel_means_ptr(voxel_means),
    voxel_covs_ptr(voxel_covs) {}

  __host__ __device__ void operator()(int i) const {
    int num_points = thrust::raw_pointer_cast(num_points_ptr)[i];
    auto& voxel_mean = thrust::raw_pointer_cast(voxel_means_ptr)[i];
    auto& voxel_covs = thrust::raw_pointer_cast(voxel_covs_ptr)[i];

    voxel_mean /= num_points;
    voxel_covs /= num_points;
  }

  thrust::device_ptr<int> num_points_ptr;
  thrust::device_ptr<Eigen::Vector3f> voxel_means_ptr;
  thrust::device_ptr<Eigen::Matrix3f> voxel_covs_ptr;
};

GaussianVoxelMapGPU::GaussianVoxelMapGPU(
  float resolution,
  int init_num_buckets,
  int max_bucket_scan_count,
  double target_points_drop_rate,
  CUstream_st* stream)
: stream(stream),
  init_num_buckets(init_num_buckets),
  target_points_drop_rate(target_points_drop_rate) {
  voxelmap_info.num_voxels = 0;
  voxelmap_info.num_buckets = init_num_buckets;
  voxelmap_info.max_bucket_scan_count = max_bucket_scan_count;
  voxelmap_info.voxel_resolution = resolution;

  check_error << cudaMallocAsync(&voxelmap_info_ptr, sizeof(VoxelMapInfo), stream);
  check_error << cudaMemcpyAsync(voxelmap_info_ptr, &voxelmap_info, sizeof(VoxelMapInfo), cudaMemcpyHostToDevice, stream);

  buckets = nullptr;

  num_points = nullptr;
  voxel_means = nullptr;
  voxel_covs = nullptr;
}

GaussianVoxelMapGPU::~GaussianVoxelMapGPU() {
  check_error << cudaFreeAsync(voxelmap_info_ptr, 0);
  check_error << cudaFreeAsync(buckets, 0);
  check_error << cudaFreeAsync(num_points, 0);
  check_error << cudaFreeAsync(voxel_means, 0);
  check_error << cudaFreeAsync(voxel_covs, 0);
}

void GaussianVoxelMapGPU::insert(const PointCloud& frame) {
  if (!frame.check_points_gpu() || !frame.check_covs_gpu()) {
    std::cerr << "error: GPU points/covs not allocated!!" << std::endl;
    abort();
  }

  create_bucket_table(stream, frame);

  check_error << cudaMallocAsync(&num_points, sizeof(int) * voxelmap_info.num_voxels, stream);
  check_error << cudaMallocAsync(&voxel_means, sizeof(Eigen::Vector3f) * voxelmap_info.num_voxels, stream);
  check_error << cudaMallocAsync(&voxel_covs, sizeof(Eigen::Matrix3f) * voxelmap_info.num_voxels, stream);

  check_error << cudaMemsetAsync(num_points, 0, sizeof(int) * voxelmap_info.num_voxels, stream);
  check_error << cudaMemsetAsync(voxel_means, 0, sizeof(Eigen::Vector3f) * voxelmap_info.num_voxels, stream);
  check_error << cudaMemsetAsync(voxel_covs, 0, sizeof(Eigen::Matrix3f) * voxelmap_info.num_voxels, stream);

  thrust::device_ptr<Eigen::Vector3f> points_ptr(frame.points_gpu);
  thrust::device_ptr<Eigen::Matrix3f> covs_ptr(frame.covs_gpu);

  thrust::for_each(
    thrust::cuda::par_nosync.on(stream),
    thrust::make_zip_iterator(thrust::make_tuple(points_ptr, covs_ptr)),
    thrust::make_zip_iterator(thrust::make_tuple(points_ptr + frame.size(), covs_ptr + frame.size())),
    accumulate_points_kernel(voxelmap_info_ptr, buckets, num_points, voxel_means, voxel_covs));

  thrust::for_each(
    thrust::cuda::par_nosync.on(stream),
    thrust::counting_iterator<int>(0),
    thrust::counting_iterator<int>(voxelmap_info.num_voxels),
    finalize_voxels_kernel(num_points, voxel_means, voxel_covs));

  cudaStreamSynchronize(stream);
}

void GaussianVoxelMapGPU::create_bucket_table(cudaStream_t stream, const PointCloud& frame) {
  // transform points(Vector3f) to voxel coords(Vector3i)
  Eigen::Vector3i* coords;
  check_error << cudaMallocAsync(&coords, sizeof(Eigen::Vector3i) * frame.size(), stream);
  thrust::transform(
    thrust::cuda::par_nosync.on(stream),
    thrust::device_ptr<Eigen::Vector3f>(frame.points_gpu),
    thrust::device_ptr<Eigen::Vector3f>(frame.points_gpu + frame.size()),
    coords,
    voxel_coord_kernel(voxelmap_info_ptr));

  thrust::pair<int, int>* index_buckets = nullptr;
  int* voxels_failures;
  check_error << cudaMallocAsync(&voxels_failures, sizeof(int) * 2, stream);
  check_error << cudaMemsetAsync(voxels_failures, 0, sizeof(int) * 2, stream);

  for (int num_buckets = init_num_buckets; init_num_buckets * 4; num_buckets *= 2) {
    voxelmap_info.num_buckets = num_buckets;
    check_error << cudaMemcpyAsync(voxelmap_info_ptr, &voxelmap_info, sizeof(VoxelMapInfo), cudaMemcpyHostToDevice, stream);

    check_error << cudaFreeAsync(index_buckets, stream);
    check_error << cudaMallocAsync(&index_buckets, sizeof(thrust::pair<int, int>) * num_buckets, stream);
    check_error << cudaMemsetAsync(index_buckets, -1, sizeof(thrust::pair<int, int>) * num_buckets, stream);
    check_error << cudaMemsetAsync(voxels_failures, 0, sizeof(int) * 2, stream);

    thrust::for_each(
      thrust::cuda::par_nosync.on(stream),
      thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(frame.size()),
      voxel_bucket_assignment_kernel(voxelmap_info_ptr, coords, index_buckets, voxels_failures));

    std::array<int, 2> h_voxels_failures;
    check_error << cudaMemcpyAsync(h_voxels_failures.data(), voxels_failures, sizeof(int) * 2, cudaMemcpyDeviceToHost, stream);
    check_error << cudaStreamSynchronize(stream);

    if (h_voxels_failures[1] == 0 || static_cast<double>(h_voxels_failures[1]) / frame.size() <= target_points_drop_rate) {
      voxelmap_info.num_voxels = h_voxels_failures[0];
      check_error << cudaMemcpyAsync(voxelmap_info_ptr, &voxelmap_info, sizeof(VoxelMapInfo), cudaMemcpyHostToDevice, stream);
      break;
    }
  }

  check_error << cudaFreeAsync(buckets, stream);
  check_error << cudaMallocAsync(&buckets, sizeof(VoxelBucket) * voxelmap_info.num_buckets, stream);
  thrust::transform(
    thrust::cuda::par_nosync.on(stream),
    thrust::device_ptr<thrust::pair<int, int>>(index_buckets),
    thrust::device_ptr<thrust::pair<int, int>>(index_buckets) + voxelmap_info.num_buckets,
    thrust::device_ptr<VoxelBucket>(buckets),
    voxel_coord_select_kernel(coords));

  check_error << cudaFreeAsync(coords, stream);
  check_error << cudaFreeAsync(voxels_failures, stream);
  check_error << cudaFreeAsync(index_buckets, stream);
}

std::vector<Eigen::Vector3f> download_voxel_means(const GaussianVoxelMapGPU& voxelmap, CUstream_st* stream) {
  std::vector<Eigen::Vector3f> means(voxelmap.voxelmap_info.num_voxels);
  check_error << cudaMemcpyAsync(
    means.data(),
    voxelmap.voxel_means,
    sizeof(Eigen::Vector3f) * voxelmap.voxelmap_info.num_voxels,
    cudaMemcpyDeviceToHost,
    stream);
  return means;
}

std::vector<Eigen::Matrix3f> download_voxel_covs(const GaussianVoxelMapGPU& voxelmap, CUstream_st* stream) {
  std::vector<Eigen::Matrix3f> covs(voxelmap.voxelmap_info.num_voxels);
  check_error
    << cudaMemcpyAsync(covs.data(), voxelmap.voxel_covs, sizeof(Eigen::Matrix3f) * voxelmap.voxelmap_info.num_voxels, cudaMemcpyDeviceToHost, stream);
  return covs;
}

}  // namespace gtsam_points