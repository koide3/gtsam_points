#include <gtsam_ext/cuda/async_stream.hpp>

#include <iostream>

namespace gtsam_ext {

AsyncStream::AsyncStream() {
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
}

AsyncStream::~AsyncStream() {
  cudaStreamSynchronize(stream);

  for (auto itr = resources.rbegin(); itr != resources.rend(); itr++) {
    itr->reset();
  }

  cudaStreamDestroy(stream);
}

void AsyncStream::sync() {
  cudaStreamSynchronize(stream);
}

void AsyncStream::add_resource(const std::shared_ptr<void>& resource) {
  resources.push_back(resource);
}

RegisteredMemory::RegisteredMemory(void* ptr, size_t size, unsigned int flags) {
  cudaHostRegister(ptr, size, flags);
}

RegisteredMemory::~RegisteredMemory() {
  cudaHostUnregister(ptr);
}

}  // namespace gtsam_ext