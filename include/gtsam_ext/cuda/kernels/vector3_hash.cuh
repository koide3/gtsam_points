// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <Eigen/Core>
#include <thrust/device_ptr.h>

namespace gtsam_ext {

// taken from boost/hash.hpp
inline __host__ __device__ void hash_combine(uint64_t& h, uint64_t k) {
  const uint64_t m = UINT64_C(0xc6a4a7935bd1e995);
  const int r = 47;

  k *= m;
  k ^= k >> r;
  k *= m;

  h ^= k;
  h *= m;

  h += 0xe6546b64;
}

inline __host__ __device__ bool equal(const Eigen::Vector3i& lhs, const Eigen::Vector3i& rhs) {
  return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2];
}

// compute vector3i hash
inline __host__ __device__ uint64_t vector3i_hash(const Eigen::Vector3i& x) {
  uint64_t seed = 0;
  hash_combine(seed, x[0]);
  hash_combine(seed, x[1]);
  hash_combine(seed, x[2]);
  return seed;
}

// real vector -> voxel index vector
inline __host__ __device__ Eigen::Vector3i calc_voxel_coord(const Eigen::Vector3f& x, float resolution) {
  Eigen::Vector3i coord = (x.array() / resolution - 0.5).floor().cast<int>();
  return coord;
}

// find corresponding voxel
inline __host__ __device__ int lookup_voxel(
  const int max_bucket_scan_count,
  const int num_buckets,
  const thrust::device_ptr<const thrust::pair<Eigen::Vector3i, int>>& buckets_ptr,
  const float resolution,
  const Eigen::Vector3f& x) {
  Eigen::Vector3i coord = calc_voxel_coord(x, resolution);
  uint64_t hash = vector3i_hash(coord);

  for (int i = 0; i < max_bucket_scan_count; i++) {
    uint64_t bucket_index = (hash + i) % num_buckets;
    const auto& bucket = thrust::raw_pointer_cast(buckets_ptr)[bucket_index];

    if (bucket.second < 0) {
      return -1;
    }

    if (equal(bucket.first, coord)) {
      return bucket.second;
    }
  }

  return -1;
}
}  // namespace gtsam_ext
