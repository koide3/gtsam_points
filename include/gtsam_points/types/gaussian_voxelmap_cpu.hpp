// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/types/gaussian_voxelmap.hpp>
#include <gtsam_points/ann/incremental_voxelmap.hpp>

namespace gtsam_points {

/// @brief Gaussian voxel that computes and stores voxel mean and covariance.
struct GaussianVoxel {
public:
  using Ptr = std::shared_ptr<GaussianVoxel>;
  using ConstPtr = std::shared_ptr<const GaussianVoxel>;

  struct Setting {};

  /// @brief Constructor.
  GaussianVoxel() : finalized(false), num_points(0), mean(Eigen::Vector4d::Zero()), cov(Eigen::Matrix4d::Zero()) {}

  /// @brief  Number of points in the voxel (Always 1 for GaussianVoxel).
  size_t size() const { return 1; }

  /// @brief  Add a point to the voxel.
  /// @param  setting        Setting
  /// @param  transformed_pt Transformed point mean
  /// @param  points         Point cloud
  /// @param  i              Index of the point
  /// @param  T              Transformation matrix
  void add(const Setting& setting, const PointCloud& points, size_t i);

  /// @brief Finalize the voxel mean and covariance.
  void finalize();

  /// @brief Find k nearest neighbors.
  /// @param pt           Query point
  /// @param result       Result
  template <typename Result>
  void knn_search(const Eigen::Vector4d& pt, Result& result) const {
    const double sq_dist = (pt - mean).squaredNorm();
    result.push(0, sq_dist);
  }

public:
  bool finalized;        ///< If true, mean and cov are finalized, otherwise they represent the sum of input points
  size_t num_points;     ///< Number of input points
  Eigen::Vector4d mean;  ///< Mean
  Eigen::Matrix4d cov;   ///< Covariance
};

namespace frame {

template <>
struct traits<GaussianVoxel> {
  static int size(const GaussianVoxel& frame) { return frame.size(); }

  static bool has_points(const GaussianVoxel& frame) { return true; }
  static bool has_normals(const GaussianVoxel& frame) { return false; }
  static bool has_covs(const GaussianVoxel& frame) { return true; }
  static bool has_intensities(const GaussianVoxel& frame) { return false; }

  static const Eigen::Vector4d& point(const GaussianVoxel& frame, size_t i) { return frame.mean; }
  static const Eigen::Vector4d normal(const GaussianVoxel& frame, size_t i) { return Eigen::Vector4d::Zero(); }
  static const Eigen::Matrix4d& cov(const GaussianVoxel& frame, size_t i) { return frame.cov; }
  static double intensity(const GaussianVoxel& frame, size_t i) { return 0.0; }
};

}  // namespace frame

class GaussianVoxelMapCPU : public GaussianVoxelMap, public IncrementalVoxelMap<GaussianVoxel> {
public:
  using Ptr = std::shared_ptr<GaussianVoxelMapCPU>;
  using ConstPtr = std::shared_ptr<const GaussianVoxelMapCPU>;

  /// @brief Constructor.
  /// @param resolution   Voxel resolution
  GaussianVoxelMapCPU(double resolution);
  virtual ~GaussianVoxelMapCPU();

  /// Voxel resolution
  virtual double voxel_resolution() const override;

  /// @brief Compute the voxel index corresponding to a point.
  Eigen::Vector3i voxel_coord(const Eigen::Vector4d& x) const;

  /// @brief Look up a voxel index. If the voxel does not exist, return -1.
  int lookup_voxel_index(const Eigen::Vector3i& coord) const;

  /// @brief Look up a voxel.
  const GaussianVoxel& lookup_voxel(int voxel_id) const;

  /// @brief  Insert a point cloud frame into the voxelmap.
  virtual void insert(const PointCloud& frame) override;

  /**
   * @brief Save the voxelmap
   * @param path  Destination path to save the voxelmap
   */
  void save_compact(const std::string& path) const;

  /**
   * @brief Save a voxelmap from a file
   * @param path  Path to a voxelmap file to be loaded
   */
  static GaussianVoxelMapCPU::Ptr load(const std::string& path);
};

namespace frame {

template <>
struct traits<GaussianVoxelMapCPU> {
  static bool has_points(const GaussianVoxelMapCPU& ivox) { return ivox.has_points(); }
  static bool has_normals(const GaussianVoxelMapCPU& ivox) { return ivox.has_normals(); }
  static bool has_covs(const GaussianVoxelMapCPU& ivox) { return ivox.has_covs(); }
  static bool has_intensities(const GaussianVoxelMapCPU& ivox) { return ivox.has_intensities(); }

  static decltype(auto) point(const GaussianVoxelMapCPU& ivox, size_t i) { return ivox.point(i); }
  static decltype(auto) normal(const GaussianVoxelMapCPU& ivox, size_t i) { return ivox.normal(i); }
  static decltype(auto) cov(const GaussianVoxelMapCPU& ivox, size_t i) { return ivox.cov(i); }
  static decltype(auto) intensity(const GaussianVoxelMapCPU& ivox, size_t i) { return ivox.intensity(i); }
};

}  // namespace frame

}  // namespace gtsam_points
