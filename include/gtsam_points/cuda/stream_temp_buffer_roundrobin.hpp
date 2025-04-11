// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <deque>
#include <memory>
#include <unordered_map>

#include <gtsam_points/cuda/stream_roundrobin.hpp>

namespace gtsam_points {

/**
 * @brief Temporary buffer manager
 * @note  This class allocates a new buffer only when a buffer larger than the largest one among allocated buffers
 * @note  This is useful for managing temporary buffers when using CUB's reduce function several times on a single CUDA stream
 */
class TempBufferManager {
public:
  struct Buffer {
    Buffer(size_t size);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    size_t size;
    char* buffer;
  };

  using Ptr = std::shared_ptr<TempBufferManager>;

  TempBufferManager(size_t init_buffer_size = 0);
  ~TempBufferManager();

  char* get_buffer(size_t buffer_size);

  void clear();
  void clear_all();

private:
  std::vector<std::shared_ptr<Buffer>> buffers;
};

/**
 * @brief Roundrobin for pairs of CUDA stream and temporary buffer manager
 */
class StreamTempBufferRoundRobin {
public:
  StreamTempBufferRoundRobin(int num_streams = 4, size_t init_buffer_size = 512 * 1024);
  ~StreamTempBufferRoundRobin();

  std::pair<CUstream_st*, TempBufferManager::Ptr> get_stream_buffer();

  void sync_all();

  void clear();
  void clear_all();

private:
  size_t init_buffer_size;
  std::unique_ptr<StreamRoundRobin> stream_roundrobin;
  std::unordered_map<CUstream_st*, TempBufferManager::Ptr> buffer_map;
};

}  // namespace gtsam_points