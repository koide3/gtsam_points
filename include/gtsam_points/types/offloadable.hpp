// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <vector>
#include <atomic>
#include <cstdint>

// forward declaration
struct CUstream_st;

namespace gtsam_points {

/**
 * @brief An interface class for offloading data from GPU to CPU.
 */
class OffloadableGPU {
public:
  using Ptr = std::shared_ptr<OffloadableGPU>;
  using ConstPtr = std::shared_ptr<const OffloadableGPU>;

  OffloadableGPU();
  virtual ~OffloadableGPU();

  /// @brief Current global access time counter
  static std::uint64_t current_access_time();

  /// @brief Time of the last access to this object
  std::uint64_t last_accessed_time() const;

  /// @brief Memory usage in bytes on the GPU
  virtual size_t memory_usage_gpu() const { return 0; }

  /// @brief Check if the data is loaded on the GPU
  virtual bool loaded_on_gpu() const = 0;

  /// @brief Reload data from CPU to GPU (if necessary) and update the last access time
  /// @return true if the data offload is conducted, false if the data is already on the CPU
  virtual bool touch(CUstream_st* stream = 0);

  /// @brief Offload data from GPU to CPU
  /// @return true if the data offload is conducted, false if the data is already on the CPU
  virtual bool offload_gpu(CUstream_st* stream = 0) = 0;

  /// @brief  Reload data from CPU to GPU
  /// @return true if the data upload is conducted, false if the data is already on the GPU
  virtual bool reload_gpu(CUstream_st* stream = 0) = 0;

protected:
  static std::atomic_uint64_t access_counter;  ///< Global counter for the last access time
  std::uint64_t last_access;                   ///< Last access time of this object
};

}  // namespace gtsam_points
