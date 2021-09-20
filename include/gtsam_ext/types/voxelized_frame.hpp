#pragma once

#include <gtsam_ext/types/frame.hpp>

#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_gpu.hpp>

namespace gtsam_ext {

// class GaussianVoxelMapCPU;
// class GaussianVoxelMapGPU;

struct VoxelizedFrame : public Frame {
public:
  using Ptr = std::shared_ptr<VoxelizedFrame>;
  using ConstPtr = std::shared_ptr<const VoxelizedFrame>;

  VoxelizedFrame() : voxels(nullptr), voxels_gpu(nullptr) {}
  virtual ~VoxelizedFrame() {}

public:
  GaussianVoxelMapCPU* voxels;
  GaussianVoxelMapGPU* voxels_gpu;
};

}  // namespace gtsam_ext
