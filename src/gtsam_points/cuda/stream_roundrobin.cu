// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/cuda/stream_roundrobin.hpp>

#include <cuda_runtime_api.h>

namespace gtsam_points {

StreamRoundRobin::StreamRoundRobin(int num_streams) {
  streams.resize(num_streams);
  for (int i = 0; i < num_streams; i++) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
  }
  cursor = 0;
}

StreamRoundRobin::~StreamRoundRobin() {
  for (int i = 0; i < streams.size(); i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }
}

void StreamRoundRobin::sync_all() {
  for (int i = 0; i < streams.size(); i++) {
    cudaStreamSynchronize(streams[i]);
  }
}

cudaStream_t StreamRoundRobin::get_stream() {
  int stream = cursor++;
  return streams[stream % streams.size()];
}

}  // namespace gtsam_points