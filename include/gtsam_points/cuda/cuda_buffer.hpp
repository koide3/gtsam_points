// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <string>

struct CUstream_st;

namespace gtsam_points {

/**
 * @brief Device buffer for asynchronous data transfer.
 * @note  To enable asynchronous upload/download, use_pinned_buffer needs to be true.
 */
class CUDABuffer {
public:
  CUDABuffer(bool use_pinned_buffer = true);
  ~CUDABuffer();

  /**
   * @brief Resize the buffer size. This method only expands the device/host buffers and
   *        doesn't shrink them when buffer_size < size.
   * @param size   Buffer size
   * @param stream CUDA stream
   */
  void resize(size_t size, CUstream_st* stream);

  /**
   * @brief Upload data from the host pinned buffer to the device buffer.
   * @param stream CUDA stream
   */
  void upload(CUstream_st* stream);

  /**
   * @brief Upload data from the host pinned buffer to the device buffer.
   * @param size   Data size (must be smaller than buffer_size)
   * @param stream CUDA stream
   */
  void upload(size_t size, CUstream_st* stream);

  /**
   * @brief Upload data to the device buffer. If size > buffer_size,
   *        the buffers will be resized before uploading.
   * @param buffer  Input data
   * @param size    Buffer size
   * @param stream  CUDA stream
   */
  void upload(const void* buffer, size_t size, CUstream_st* stream);

  /**
   * @brief Download data from the device buffer to the pinned host buffer.
   * @param stream  CUDA stream
   */
  void download(CUstream_st* stream);

  /**
   * @brief Download data from the device buffer.
   * @param buffer  Buffer to write the downloaded data
   * @param size    Buffer size
   * @param stream  CUDA stream
   */
  void download(void* buffer, size_t size, CUstream_st* stream);

  /**
   * @brief Buffer size.
   */
  size_t size() const;

  /**
   * @brief Pinned host buffer.
   * @note  If use_pinned_buffer is false, the pinned host buffer
   *        will not be allocated, and this method returns nullptr.
   */
  void* host_buffer();

  /**
   * @brief Pinned device buffer.
   */
  void* device_buffer();

  template <typename T>
  void upload(const T* buffer, size_t size, CUstream_st* stream) {
    upload(reinterpret_cast<const void*>(buffer), size, stream);
  }
  template <typename T>
  void download(T* buffer, size_t size, CUstream_st* stream) {
    download(reinterpret_cast<void*>(buffer), size, stream);
  }
  template <typename T>
  T* host_buffer() {
    return reinterpret_cast<T*>(host_buffer());
  }
  template <typename T>
  T* device_buffer() {
    return reinterpret_cast<T*>(device_buffer());
  }

private:
  const bool use_pinned_buffer;

  size_t buffer_size;
  void* h_buffer;
  void* d_buffer;
};

}  // namespace gtsam_points
