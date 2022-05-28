// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <atomic>
#include <memory>
#include <unordered_map>

#include <Eigen/Core>
#include <gtsam_ext/util/vector3i_hash.hpp>
#include <gtsam_ext/types/gaussian_voxelmap.hpp>

namespace gtsam_ext {

/**
 * @brief Gaussian distribution voxel
 */
struct GaussianVoxel {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<GaussianVoxel>;
  using ConstPtr = std::shared_ptr<const GaussianVoxel>;

  GaussianVoxel();
  ~GaussianVoxel();

  /**
   * @brief Add a point to the voxel distribution.
   * \note  finalize() must be called before using calculated mean and cov.
   */
  void append(const Eigen::Vector4d& mean_, const Eigen::Matrix4d& cov_);

  /**
   * @brief Finalize the distribution parameters (mean and cov).
   * \note  You can add more points and re-finalize the distribution parameters over and over.
   */
  void finalize();

public:
  bool finalized;                          ///< If the mean and cov are finalized
  mutable std::atomic_int last_lru_count;  ///< LRU cache count

  int num_points;        ///< Number of inserted points
  Eigen::Vector4d mean;  ///< Mean. If it is not finalized, it stores num_points * mean instead;
  Eigen::Matrix4d cov;   ///< Covariance. If it is not finalized, it stores num_points * cov instead;
};

/**
 * @brief Gaussian distribution voxelmap
 */
class GaussianVoxelMapCPU : public GaussianVoxelMap {
public:
  using Ptr = std::shared_ptr<GaussianVoxelMapCPU>;
  using ConstPtr = std::shared_ptr<const GaussianVoxelMapCPU>;

  /**
   * @brief Constructor
   * @param resolution Voxel resolution
   */
  GaussianVoxelMapCPU(double resolution);
  virtual ~GaussianVoxelMapCPU();

  /// LRU clearing check cycle
  void set_lru_cycle(const int cycle) { lru_cycle = cycle; }

  /// LRU cache threshold
  void set_lru_thresh(const int thresh) { lru_thresh = thresh; }

  /// Voxel resolution
  virtual double voxel_resolution() const override { return resolution; }

  /**
   * @brief Insert a point cloud frame into the voxelmap
   * @param frame Point cloud frame to be inserted
   */
  virtual void insert(const Frame& frame) override;

  /**
   * @brief Calculate the coordinates of a voxel that contains $x$.
   * @return Calculated voxel coordinates
   */
  Eigen::Vector3i voxel_coord(const Eigen::Vector4d& x) const;

  /**
   * @brief Look up a voxel
   * @param coord Voxel coordinates
   */
  GaussianVoxel::Ptr lookup_voxel(const Eigen::Vector3i& coord) const;

public:
  using VoxelMap = std::unordered_map<
    Eigen::Vector3i,
    GaussianVoxel::Ptr,
    XORVector3iHash,
    std::equal_to<Eigen::Vector3i>,
    Eigen::aligned_allocator<std::pair<const Eigen::Vector3i, GaussianVoxel::Ptr>>>;

  int lru_count;   ///< LRU count
  int lru_cycle;   ///< LRU check cycle
  int lru_thresh;  ///< LRU threshold. Voxels that are not observed longer than this are removed from the voxelmap.

  double resolution;  ///< Voxel resolution
  VoxelMap voxels;    ///< Voxelmap
};

}  // namespace gtsam_ext
