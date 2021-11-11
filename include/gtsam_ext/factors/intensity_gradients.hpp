#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>

#include <gtsam_ext/types/frame.hpp>
#include <gtsam_ext/types/frame_cpu.hpp>

namespace gtsam_ext {

class IntensityGradients {
public:
  using Ptr = std::shared_ptr<IntensityGradients>;
  using ConstPtr = std::shared_ptr<const IntensityGradients>;

  static IntensityGradients::Ptr estimate(const gtsam_ext::Frame::ConstPtr& frame, int k_neighbors = 10);
  static IntensityGradients::Ptr estimate(const gtsam_ext::FrameCPU::Ptr& frame, int k_neighbors = 10);
  static IntensityGradients::Ptr estimate(const gtsam_ext::FrameCPU::Ptr& frame, int k_geom_neighbors, int k_photo_neighbors);

public:
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> intensity_gradients;
};

}