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

  Frame()
  : num_points(0),
    times(nullptr),
    points(nullptr),
    normals(nullptr),
    covs(nullptr),
    intensities(nullptr),
    times_gpu(nullptr),
    points_gpu(nullptr),
    normals_gpu(nullptr),
    covs_gpu(nullptr),
    intensities_gpu(nullptr) {}
  virtual ~Frame() {}

  size_t size() const { return num_points; }

  // Check if the frame has attributes
  bool has_times() const;
  bool has_points() const;
  bool has_normals() const;
  bool has_covs() const;
  bool has_intensities() const;

  // calculate the fraction of points fell in target's voxels
  // (check if delta * this->points fall in target->voxels)
  double overlap(const std::shared_ptr<const VoxelizedFrame>& target, const Eigen::Isometry3d& delta) const;
  double overlap(const std::vector<std::shared_ptr<const VoxelizedFrame>>& targets, const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas)
    const;

  double overlap_gpu(const std::shared_ptr<const VoxelizedFrame>& target, const Eigen::Isometry3f* delta_gpu) const;
  double overlap_gpu(const std::shared_ptr<const VoxelizedFrame>& target, const Eigen::Isometry3d& delta) const;
  double overlap_gpu(const std::vector<std::shared_ptr<const VoxelizedFrame>>& targets, const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas)
    const;

  // Automatically select CPU or GPU method
  double overlap_auto(const std::shared_ptr<const VoxelizedFrame>& target, const Eigen::Isometry3d& delta) const;
  double overlap_auto(const std::vector<std::shared_ptr<const VoxelizedFrame>>& targets, const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas)
    const;

  void save(const std::string& path) const;
  void save_compact(const std::string& path) const;

public:
  size_t num_points;

  double* times;             // time w.r.t. the first point (sorted)
  Eigen::Vector4d* points;   // point coordinates (x, y, z, 1)
  Eigen::Vector4d* normals;  // point normals (nx, ny, nz, 0)
  Eigen::Matrix4d* covs;     // point covariances cov(3, 3) = 0
  double* intensities;       // point intensity

  float* times_gpu;
  Eigen::Vector3f* points_gpu;
  Eigen::Vector3f* normals_gpu;
  Eigen::Matrix3f* covs_gpu;
  float* intensities_gpu;
};

}  // namespace gtsam_ext