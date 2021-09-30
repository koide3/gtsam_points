#pragma once

#include <vector>
#include <Eigen/Core>

#include <gtsam_ext/types/frame.hpp>

namespace gtsam_ext {

struct FrameCPU : public Frame {
public:
  using Ptr = std::shared_ptr<FrameCPU>;
  using ConstPtr = std::shared_ptr<const FrameCPU>;

  template <typename T, int D>
  FrameCPU(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& points);

  template <typename T, int D>
  FrameCPU(
    const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& points,
    const std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>>& covs);

  ~FrameCPU();

public:
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points_storage;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> covs_storage;
};

}  // namespace gtsam_ext
