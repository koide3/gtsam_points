// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/cuda/cuda_buffer.hpp>

#include <gtsam_points/cuda/check_error.cuh>

namespace gtsam_points {

CUDABuffer::CUDABuffer(bool use_pinned_buffer) : use_pinned_buffer(use_pinned_buffer), buffer_size(0), h_buffer(nullptr), d_buffer(nullptr) {}

CUDABuffer::~CUDABuffer() {
  if (h_buffer) {
    check_error << cudaFreeHost(h_buffer);
  }
  if (d_buffer) {
    check_error << cudaFreeAsync(d_buffer, 0);
  }
}

void CUDABuffer::resize(size_t size, CUstream_st* stream) {
  if (buffer_size < size) {
    size = size * 1.2;
    if (use_pinned_buffer) {
      check_error << cudaFreeHost(h_buffer);
      check_error << cudaMallocHost(&h_buffer, size);
    }

    check_error << cudaFreeAsync(d_buffer, stream);
    check_error << cudaMallocAsync(&d_buffer, size, stream);
    buffer_size = size;
  }
}

void CUDABuffer::upload(size_t size, CUstream_st* stream) {
  if (size > buffer_size) {
    std::cerr << "error: data size must be smaller than buffer_size!!" << std::endl;
    std::cerr << "     : size=" << size << " buffer_size=" << buffer_size << std::endl;
    abort();
  }

  if (!use_pinned_buffer) {
    std::cerr << "error: pinned buffer must be enabled!!" << std::endl;
    abort();
  }

  check_error << cudaMemcpyAsync(d_buffer, h_buffer, size, cudaMemcpyHostToDevice, stream);
}

void CUDABuffer::upload(CUstream_st* stream) {
  upload(buffer_size, stream);
}

void CUDABuffer::upload(const void* buffer, size_t size, CUstream_st* stream) {
  resize(size, stream);

  const void* src_buffer = buffer;
  if (use_pinned_buffer) {
    check_error << cudaMemcpyAsync(h_buffer, buffer, size, cudaMemcpyHostToHost, stream);
    src_buffer = h_buffer;
  }

  check_error << cudaMemcpyAsync(d_buffer, src_buffer, size, cudaMemcpyHostToDevice, stream);
}

void CUDABuffer::download(CUstream_st* stream) {
  if (!use_pinned_buffer) {
    std::cerr << "error: trying to dowload data to disabled pinned host buffer!!" << std::endl;
    abort();
  }

  check_error << cudaMemcpyAsync(h_buffer, d_buffer, buffer_size, cudaMemcpyDeviceToHost, stream);
}

void CUDABuffer::download(void* buffer, size_t size, CUstream_st* stream) {
  if (use_pinned_buffer) {
    check_error << cudaMemcpyAsync(h_buffer, d_buffer, buffer_size, cudaMemcpyDeviceToHost, stream);
    check_error << cudaMemcpyAsync(buffer, h_buffer, size, cudaMemcpyHostToHost, stream);
  } else {
    check_error << cudaMemcpyAsync(buffer, d_buffer, size, cudaMemcpyDeviceToHost, stream);
  }
  check_error << cudaStreamSynchronize(stream);
}

size_t CUDABuffer::size() const {
  return buffer_size;
}

void* CUDABuffer::host_buffer() {
  return h_buffer;
}

void* CUDABuffer::device_buffer() {
  return d_buffer;
}

}  // namespace gtsam_points
