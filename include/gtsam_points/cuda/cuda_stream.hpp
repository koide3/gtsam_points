// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)
#pragma once

#include <memory>
#include <vector>

struct CUstream_st;

namespace gtsam_points {

struct CUDAStream {
public:
  CUDAStream();
  ~CUDAStream();

  CUDAStream(const CUDAStream&) = delete;
  CUDAStream& operator=(const CUDAStream&) = delete;

  operator CUstream_st*() const { return stream; }

  void sync();

  void add_resource(const std::shared_ptr<void>& resource);

  template <typename T, class... Args>
  void emplace_resource(Args&&... args) {
    resources.emplace_back(std::make_shared<T>(args...));
  }

public:
  CUstream_st* stream;
  std::vector<std::shared_ptr<void>> resources;
};

struct RegisteredMemory {
public:
  RegisteredMemory(void* ptr, size_t size, unsigned int flags = 0x00);
  ~RegisteredMemory();

  RegisteredMemory(const RegisteredMemory&) = delete;
  RegisteredMemory& operator=(const RegisteredMemory&) = delete;

  void* ptr;
};

}  // namespace gtsam_points