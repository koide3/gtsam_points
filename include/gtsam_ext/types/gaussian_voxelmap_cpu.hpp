#pragma once

#include <memory>
#include <unordered_map>

#include <Eigen/Core>
#include <gtsam_ext/types/gaussian_voxelmap.hpp>

namespace gtsam_ext {

class Vector3iHash {
public:
  size_t operator()(const Eigen::Vector3i& x) const;
};

struct GaussianVoxel {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<GaussianVoxel>;
  using ConstPtr = std::shared_ptr<const GaussianVoxel>;

  GaussianVoxel();

  void append(const Eigen::Vector4d& mean_, const Eigen::Matrix4d& cov_);
  void finalize();

public:
  int num_points;
  Eigen::Vector4d mean;
  Eigen::Matrix4d cov;
};

class GaussianVoxelMapCPU : public GaussianVoxelMap {
public:
  using Ptr = std::shared_ptr<GaussianVoxelMapCPU>;
  using ConstPtr = std::shared_ptr<const GaussianVoxelMapCPU>;

  GaussianVoxelMapCPU(double resolution);
  virtual ~GaussianVoxelMapCPU();

  virtual double voxel_resolution() const override { return resolution; }
  virtual void create_voxelmap(const Frame& frame) override;

  Eigen::Vector3i voxel_coord(const Eigen::Vector4d& x) const;
  GaussianVoxel::Ptr lookup_voxel(const Eigen::Vector3i& coord) const;

public:
  using VoxelMap = std::unordered_map<
    Eigen::Vector3i,
    GaussianVoxel::Ptr,
    Vector3iHash,
    std::equal_to<Eigen::Vector3i>,
    Eigen::aligned_allocator<std::pair<const Eigen::Vector3i, GaussianVoxel::Ptr>>>;

  double resolution;
  VoxelMap voxels;
};

}  // namespace gtsam_ext
