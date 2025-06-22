#pragma once

#include <vector>
#include <atomic>
#include <cstdint>

// forward declaration
struct CUstream_st;

namespace gtsam_points {

/**
 * @brief An interface class for offloading data on the GPU memory.
 */
class OffloadableGPU {
public:
  OffloadableGPU();
  virtual ~OffloadableGPU();

  // GPU memory offloading
  static std::uint64_t current_access_time();
  std::uint64_t last_accessed_time() const;

  virtual size_t memory_usage_gpu() const { return 0; }

  virtual bool touch(CUstream_st* stream = 0) = 0;
  virtual bool offload_gpu(CUstream_st* stream = 0) = 0;
  virtual bool reload_gpu(CUstream_st* stream = 0) = 0;

private:
  static std::atomic_uint64_t access_counter;  ///< Counter for the last access time
  std::uint64_t last_access;
};

}  // namespace gtsam_points
