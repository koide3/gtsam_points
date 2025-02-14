// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>

#include <memory>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_points/factors/integrated_matching_cost_factor.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>

namespace gtsam_points {

struct GaussianVoxel;

/**
 * @brief Voxelized GICP matching cost factor
 *        Koide et al., "Voxelized GICP for Fast and Accurate 3D Point Cloud Registration", ICRA2021
 *        Koide et al., "Globally Consistent 3D LiDAR Mapping with GPU-accelerated GICP Matching Cost Factors", RA-L2021
 */
template <typename SourceFrame = gtsam_points::PointCloud>
class IntegratedVGICPFactor_ : public gtsam_points::IntegratedMatchingCostFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedVGICPFactor_>;

  /**
   * @brief Create a binary VGICP factor between target and source poses.
   * @param target_key          Target key
   * @param source_key          Source key
   * @param target_voxels       Target voxelmap
   * @param source              Source point cloud frame
   */
  IntegratedVGICPFactor_(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const GaussianVoxelMap::ConstPtr& target_voxels,
    const std::shared_ptr<const SourceFrame>& source);

  /**
   * @brief Create a unary VGICP factor between a fixed target pose and an active source pose.
   * @param fixed_target_pose   Fixed target pose
   * @param source_key          Source key
   * @param target              Target voxelized point cloud frame
   * @param source              Source point cloud frame
   */
  IntegratedVGICPFactor_(
    const gtsam::Pose3& fixed_target_pose,
    gtsam::Key source_key,
    const GaussianVoxelMap::ConstPtr& target_voxels,
    const std::shared_ptr<const SourceFrame>& source);

  virtual ~IntegratedVGICPFactor_() override;

  /// @brief Print the factor information.
  virtual void print(const std::string& s = "", const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const override;

  /// @brief Set the number of thread used for linearization of this factor.
  /// @note If your GTSAM is built with TBB, linearization is already multi-threaded
  ///       and setting n>1 can rather affect the processing speed.
  void set_num_threads(int n) { num_threads = n; }

  /// @brief Set the cache mode for fused covariance matrices (i.e., mahalanobis).
  void set_fused_cov_cache_mode(FusedCovCacheMode mode) { mahalanobis_cache_mode = mode; }

  /// @brief  Get the number of inlier points.
  /// @note   This function must be called after the factor is linearized.
  int num_inliers() const {
    const int outliers = std::count(correspondences.begin(), correspondences.end(), nullptr);
    return correspondences.size() - outliers;
  }

  /// @brief Compute the fraction of inlier points that have correspondences fell in a voxel.
  /// @note  This function must be called after the factor is linearized.
  double inlier_fraction() const { return num_inliers() / static_cast<double>(correspondences.size()); }

  /// @brief Get the target voxelmap.
  const std::shared_ptr<const GaussianVoxelMapCPU>& get_target() const { return target_voxels; }

  gtsam::NonlinearFactor::shared_ptr clone() const override { return gtsam::NonlinearFactor::shared_ptr(new IntegratedVGICPFactor_(*this)); }

private:
  virtual void update_correspondences(const Eigen::Isometry3d& delta) const override;

  virtual double evaluate(
    const Eigen::Isometry3d& delta,
    Eigen::Matrix<double, 6, 6>* H_target = nullptr,
    Eigen::Matrix<double, 6, 6>* H_source = nullptr,
    Eigen::Matrix<double, 6, 6>* H_target_source = nullptr,
    Eigen::Matrix<double, 6, 1>* b_target = nullptr,
    Eigen::Matrix<double, 6, 1>* b_source = nullptr) const override;

private:
  int num_threads;
  FusedCovCacheMode mahalanobis_cache_mode;

  // I'm unhappy to have mutable members...
  mutable Eigen::Isometry3d linearization_point;
  mutable std::vector<const GaussianVoxel*> correspondences;
  mutable std::vector<Eigen::Matrix4d> mahalanobis_full;
  mutable std::vector<Eigen::Matrix<float, 6, 1>> mahalanobis_compact;

  std::shared_ptr<const GaussianVoxelMapCPU> target_voxels;
  std::shared_ptr<const SourceFrame> source;
};

using IntegratedVGICPFactor = IntegratedVGICPFactor_<>;

}  // namespace gtsam_points