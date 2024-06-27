// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/cuda/cuda_stream.hpp>

#include <iostream>
#include <gtsam_points/cuda/check_error.cuh>

namespace gtsam_points {

CUDAStream::CUDAStream() {
  check_error << cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
}

CUDAStream::~CUDAStream() {
  check_error << cudaStreamSynchronize(stream);

  for (auto itr = resources.rbegin(); itr != resources.rend(); itr++) {
    itr->reset();
  }

  check_error << cudaStreamDestroy(stream);
}

void CUDAStream::sync() {
  check_error << cudaStreamSynchronize(stream);
}

void CUDAStream::add_resource(const std::shared_ptr<void>& resource) {
  resources.push_back(resource);
}

RegisteredMemory::RegisteredMemory(void* ptr, size_t size, unsigned int flags) {
  check_error << cudaHostRegister(ptr, size, flags);
}

RegisteredMemory::~RegisteredMemory() {
  check_error << cudaHostUnregister(ptr);
}

}  // namespace gtsam_points