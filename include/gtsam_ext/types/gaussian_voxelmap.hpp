#pragma once

#include <gtsam_ext/types/frame.hpp>

namespace gtsam_ext {

class GaussianVoxelMap {
public:
  GaussianVoxelMap() {}
  virtual ~GaussianVoxelMap() {}

  virtual double voxel_resolution() const = 0;
  virtual void create_voxelmap(const Frame& frame) = 0;
};

}  // namespace gtsam_ext