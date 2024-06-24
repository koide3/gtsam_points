// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <thrust/pair.h>

namespace gtsam_points {

template <typename T1, typename T2>
struct untie_pair_first {
  __device__ T1 operator()(const thrust::pair<T1, T2>& x) const { return x.first; }
};

template <typename T1, typename T2>
struct untie_pair_second {
  __device__ T1 operator()(const thrust::pair<T1, T2>& x) const { return x.second; }
};

template <typename Tuple, int N>
struct untie_tuple {
  __device__ auto operator()(const Tuple& tuple) const -> decltype(thrust::get<N>(tuple)) { return thrust::get<N>(tuple); }
};

}  // namespace gtsam_points