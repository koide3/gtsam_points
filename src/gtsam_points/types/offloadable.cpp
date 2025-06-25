// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/types/offloadable.hpp>

namespace gtsam_points {

std::atomic_uint64_t OffloadableGPU::access_counter(0);  ///< Counter for the last access time

OffloadableGPU::OffloadableGPU() : last_access(access_counter.load()) {}

OffloadableGPU::~OffloadableGPU() {}

// GPU memory offloading
std::uint64_t OffloadableGPU::current_access_time() {
  return access_counter.load();
}

std::uint64_t OffloadableGPU::last_accessed_time() const {
  return last_access;
}

bool OffloadableGPU::touch(CUstream_st* stream) {
  last_access = (access_counter++);
  return reload_gpu(stream);
}

}  // namespace gtsam_points
