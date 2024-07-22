// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)
#pragma once

#include <memory>
#include <iostream>

struct CUstream_st;

namespace gtsam_points {

// Simple wrapper functions to manipulate objects on GPU memory from non-CUDA code

void* cuda_malloc(size_t size, CUstream_st* stream = nullptr);

void* cuda_malloc_and_upload(const void* data, size_t size, CUstream_st* stream = nullptr);

void cuda_free(void* ptr, CUstream_st* stream = nullptr);

void cuda_host_to_device(void* dst, const void* src, size_t size, CUstream_st* stream = nullptr);

void cuda_device_to_host(void* dst, const void* src, size_t size, CUstream_st* stream = nullptr);

void cuda_mem_get_info(size_t* free, size_t* total);

template <typename T>
T* cuda_new(CUstream_st* stream = nullptr) {
  return cuda_malloc(sizeof(T), stream);
}

template <typename T>
T* cuda_new(const T& data, CUstream_st* stream = nullptr) {
  void* ptr = cuda_malloc_and_upload(reinterpret_cast<const void*>(&data), sizeof(T), stream);
  return reinterpret_cast<T*>(ptr);
}

template <typename T>
void cuda_delete(T* ptr, CUstream_st* stream = nullptr) {
  cuda_free(reinterpret_cast<void*>(ptr), stream);
}

template <typename T>
void cuda_host_to_device(T* dst, const T* src, size_t size, CUstream_st* stream = nullptr) {
  cuda_host_to_device(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), size, stream);
}

template <typename T>
void cuda_device_to_host(T* dst, const T* src, size_t size, CUstream_st* stream = nullptr) {
  cuda_device_to_host(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), size, stream);
}

template <typename T>
std::shared_ptr<T> cuda_make_shared(CUstream_st* stream = nullptr) {
  T* ptr = reinterpret_cast<T*>(cuda_malloc(sizeof(T), stream));
  return std::shared_ptr<T>(ptr, [](T* ptr) { cuda_free(ptr); });
}

template <typename T>
std::shared_ptr<T> cuda_make_shared(const T& data, CUstream_st* stream = nullptr) {
  T* ptr = reinterpret_cast<T*>(cuda_malloc_and_upload(&data, sizeof(T), stream));
  return std::shared_ptr<T>(ptr, [](T* ptr) { cuda_free(ptr); });
}

template <typename T>
auto cuda_make_unique(CUstream_st* stream = nullptr) {
  T* ptr = reinterpret_cast<T*>(cuda_malloc(sizeof(T), stream));
  const auto deleter = [](T* ptr) { cuda_free(ptr); };
  return std::unique_ptr<T, decltype(deleter)>(ptr);
}

template <typename T>
auto cuda_make_unique(const T& data, CUstream_st* stream = nullptr) {
  T* ptr = cuda_malloc_and_upload(data, stream);
  const auto deleter = [](T* ptr) { cuda_free(ptr); };
  return std::unique_ptr<T, decltype(deleter)>(ptr);
}

template <typename T>
T cuda_download(const T* ptr, CUstream_st* stream = nullptr) {
  T data;
  cuda_device_to_host(&data, ptr, sizeof(T), stream);
  return data;
}

template <typename T>
void cuda_upload(T* dst, const T* src, CUstream_st* stream = nullptr) {
  cuda_host_to_device(dst, src, sizeof(T), stream);
}

}  // namespace gtsam_points
