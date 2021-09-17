#pragma once

#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace gtsam_ext {

struct Frame {
public:
  using Ptr = std::shared_ptr<Frame>;
  using ConstPtr = std::shared_ptr<const Frame>;

  Frame() : num_points(0), points(nullptr), covs(nullptr), points_gpu(nullptr), covs_gpu(nullptr) {}
  virtual ~Frame() {}

  size_t size() const { return num_points; }

public:
  size_t num_points;

  Eigen::Vector4d* points;
  Eigen::Matrix4d* covs;

  Eigen::Vector3f* points_gpu;
  Eigen::Matrix3f* covs_gpu;
};

}  // namespace gtsam_ext