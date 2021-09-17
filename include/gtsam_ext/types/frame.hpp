#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace gtsam_ext {

struct VoxelizedFrame;

struct Frame {
public:
  using Ptr = std::shared_ptr<Frame>;
  using ConstPtr = std::shared_ptr<const Frame>;

  Frame() : num_points(0), points(nullptr), covs(nullptr), points_gpu(nullptr), covs_gpu(nullptr) {}
  virtual ~Frame() {}

  size_t size() const { return num_points; }

  // calculate the fraction of points fell in the target's voxel
  // (check if delta * this->points fall in target->voxels)
  double overlap(const std::shared_ptr<VoxelizedFrame>& target, const Eigen::Isometry3d& delta) const;
  double overlap(const std::vector<std::shared_ptr<VoxelizedFrame>>& targets, const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas) const;

  double overlap_gpu(const std::shared_ptr<VoxelizedFrame>& target, const Eigen::Isometry3f* delta_gpu) const;
  double overlap_gpu(const std::shared_ptr<VoxelizedFrame>& target, const Eigen::Isometry3d& delta) const;
  double overlap_gpu(const std::vector<std::shared_ptr<VoxelizedFrame>>& targets, const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas) const;

public:
  size_t num_points;

  Eigen::Vector4d* points;
  Eigen::Matrix4d* covs;

  Eigen::Vector3f* points_gpu;
  Eigen::Matrix3f* covs_gpu;
};

}  // namespace gtsam_ext