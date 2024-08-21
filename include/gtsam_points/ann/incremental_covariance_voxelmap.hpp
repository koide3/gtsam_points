// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <deque>
#include <gtsam_points/util/runnning_statistics.hpp>
#include <gtsam_points/ann/incremental_voxelmap.hpp>
#include <gtsam_points/ann/incremental_covariance_container.hpp>

namespace gtsam_points {

/// @brief Incremental voxelmap with online covariance and normal estimation
struct IncrementalCovarianceVoxelMap : public IncrementalVoxelMap<IncrementalCovarianceContainer> {
public:
  /// @brief Constructor.
  /// @param voxel_resolution Voxel resolution
  IncrementalCovarianceVoxelMap(double voxel_resolution);
  virtual ~IncrementalCovarianceVoxelMap();

  /// @brief Set the number of neighbors for covariance estimation.
  void set_num_neighbors(int num_neighbors);
  /// @brief Set the minimum number of neighbors for covariance estimation.
  void set_min_num_neighbors(int min_num_neighbors);
  /// @brief Set the number of warmup cycles. Covariances of new points in this period are not re-evaluated every frame.
  void set_warmup_cycles(int warmup_cycles);
  /// @brief Set the number of lowrate update cycles. Covariances of invalid points are re-evaluated every this period.
  void set_lowrate_cycles(int lowrate_cycles);
  /// @brief Set the age threshold for removing invalid points. Invalid points older than this are removed.
  void set_remove_invalid_age_thresh(int remove_invalid_age_thresh);
  /// @brief Set the threshold scale for normal validation.
  void set_eig_stddev_thresh_scale(double eig_stddev_thresh_scale);
  /// @brief Set the number of threads for normal estimation.
  void set_num_threads(int num_threads);

  /// @brief Clear the voxelmap.
  virtual void clear() override;

  /// @brief Insert point into the voxelmap.
  virtual void insert(const PointCloud& points) override;

  /// @brief Find k-nearest neighbors. This only finds neighbors with valid covariances.
  virtual size_t knn_search(
    const double* pt,
    size_t k,
    size_t* k_indices,
    double* k_sq_dists,
    double max_sq_dist = std::numeric_limits<double>::max()) const override;

  /// @brief Find k-nearest neighbors. This finds neighbors regardless of the validity of covariances.
  size_t knn_search_force(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists, double max_sq_dist = std::numeric_limits<double>::max())
    const;

  /// @brief Get valid point indices. If num_threads is -1, the member variable num_threads is used.
  std::vector<size_t> valid_indices(int num_threads = -1) const;
  /// @brief Get points from indices.
  std::vector<Eigen::Vector4d> voxel_points(const std::vector<size_t>& indices) const;
  /// @brief Get normals from indices.
  std::vector<Eigen::Vector4d> voxel_normals(const std::vector<size_t>& indices) const;
  /// @brief Get covariances from indices.
  std::vector<Eigen::Matrix4d> voxel_covs(const std::vector<size_t>& indices) const;

  /// @brief Get voxel points
  virtual std::vector<Eigen::Vector4d> voxel_points() const override;
  /// @brief Get voxel normals
  virtual std::vector<Eigen::Vector4d> voxel_normals() const override;
  /// @brief Get voxel covariances
  virtual std::vector<Eigen::Matrix4d> voxel_covs() const override;

protected:
  int num_neighbors;               ///< Number of neighbors for covariance estimation.
  int min_num_neighbors;           ///< Minimum number of neighbors for covariance estimation.
  int warmup_cycles;               ///< Number of cycles for covariance warmup.
  int lowrate_cycles;              ///< Number of cycles for lowrate covariance estimation.
  int remove_invalid_age_thresh;   ///< Age threshold for removing invalid points.
  double eig_stddev_thresh_scale;  ///< Threshold scale for normal validation.
  int num_threads;                 ///< Number of threads for normal estimation.

  std::deque<RunningStatistics<Eigen::Array3d>> eig_stats;  ///< Running statistics for eigenvalues.
};

namespace frame {

template <>
struct traits<IncrementalCovarianceVoxelMap> {
  static bool has_points(const IncrementalCovarianceVoxelMap& ivox) { return ivox.has_points(); }
  static bool has_normals(const IncrementalCovarianceVoxelMap& ivox) { return ivox.has_normals(); }
  static bool has_covs(const IncrementalCovarianceVoxelMap& ivox) { return ivox.has_covs(); }
  static bool has_intensities(const IncrementalCovarianceVoxelMap& ivox) { return ivox.has_intensities(); }

  static const Eigen::Vector4d& point(const IncrementalCovarianceVoxelMap& ivox, size_t i) { return ivox.point(i); }
  static const Eigen::Vector4d& normal(const IncrementalCovarianceVoxelMap& ivox, size_t i) { return ivox.normal(i); }
  static const Eigen::Matrix4d& cov(const IncrementalCovarianceVoxelMap& ivox, size_t i) { return ivox.cov(i); }
  static double intensity(const IncrementalCovarianceVoxelMap& ivox, size_t i) { return ivox.intensity(i); }
};

}  // namespace frame

}  // namespace gtsam_points
