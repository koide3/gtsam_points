#pragma once

#include <gtsam_ext/types/frame.hpp>

namespace gtsam_ext {

class GaussianVoxelMapCPU;
class GaussianVoxelMapGPU;

struct VoxelizedFrame : public Frame {
public:
  VoxelizedFrame() : voxels(nullptr), voxels_gpu(nullptr) {}
  virtual ~VoxelizedFrame() {}

public:
  GaussianVoxelMapCPU* voxels;
  GaussianVoxelMapGPU* voxels_gpu;
};

}  // namespace gtsam_ext
