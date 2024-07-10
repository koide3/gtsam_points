// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <atomic>
#include <vector>

struct CUstream_st;

namespace gtsam_points {

/**
 * @brief Roundrobin for CUDA streams
 */
class StreamRoundRobin {
public:
  using cudaStream_t = CUstream_st*;

  StreamRoundRobin(int num_streams);
  ~StreamRoundRobin();

  void sync_all();

  cudaStream_t get_stream();

private:
  std::atomic_int cursor;
  std::vector<cudaStream_t> streams;
};

}  // namespace gtsam_points