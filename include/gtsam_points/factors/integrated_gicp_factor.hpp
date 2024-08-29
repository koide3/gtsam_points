// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>

#include <memory>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/factors/integrated_matching_cost_factor.hpp>

namespace gtsam_points {

struct NearestNeighborSearch;

/**
 * @brief Generalized ICP matching cost factor
 *        Segal et al., "Generalized-ICP", RSS2005
 */
template <typename TargetFrame = gtsam_points::PointCloud, typename SourceFrame = gtsam_points::PointCloud>
class IntegratedGICPFactor_ : public gtsam_points::IntegratedMatchingCostFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedGICPFactor_>;

  /**
   * @brief Create a binary ICP factor between target and source poses.
   * @param target_key          Target key
   * @param source_key          Source key
   * @param target              Target point cloud frame
   * @param source              Source point cloud frame
   * @param target_tree         Target nearest neighbor search
   */
  IntegratedGICPFactor_(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source,
    const std::shared_ptr<const NearestNeighborSearch>& target_tree);

  ///< Create a binary ICP factor between target and source poses.
  IntegratedGICPFactor_(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source);

  /**
   * @brief Create a unary GICP factor between a fixed target pose and an active source pose.
   * @param fixed_target_pose   Fixed target pose
   * @param source_key          Source key
   * @param target              Target point cloud frame
   * @param source              Source point cloud frame
   * @param target_tree         Target nearest neighbor search
   */
  IntegratedGICPFactor_(
    const gtsam::Pose3& fixed_target_pose,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source,
    const std::shared_ptr<const NearestNeighborSearch>& target_tree);

  ///< Create a unary GICP factor between a fixed target pose and an active source pose.
  IntegratedGICPFactor_(
    const gtsam::Pose3& fixed_target_pose,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source);

  virtual ~IntegratedGICPFactor_() override;

  /// @brief Set the number of thread used for linearization of this factor.
  /// @note If your GTSAM is built with TBB, linearization is already multi-threaded
  ///       and setting n>1 can rather affect the processing speed.
  void set_num_threads(int n) { num_threads = n; }

  /// @brief Set the maximum distance between corresponding points.
  ///        Correspondences with distances larger than this will be rejected (i.e., correspondence trimming).
  void set_max_correspondence_distance(double dist) { max_correspondence_distance_sq = dist * dist; }

  /// @brief Correspondences are updated only when the displacement from the last update point is larger than these threshold values.
  /// @note  Default values are angle=trans=0 and correspondences are updated every linearization call.
  void set_correspondence_update_tolerance(double angle, double trans) {
    correspondence_update_tolerance_rot = angle;
    correspondence_update_tolerance_trans = trans;
  }

  /// @brief Compute the fraction of inlier points that have correspondences with a distance smaller than the trimming threshold.
  double inlier_fraction() const {
    const int outliers = std::count(correspondences.begin(), correspondences.end(), -1);
    const int inliers = correspondences.size() - outliers;
    return static_cast<double>(inliers) / correspondences.size();
  }

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
  double max_correspondence_distance_sq;

  std::shared_ptr<const NearestNeighborSearch> target_tree;

  // I'm unhappy to have mutable members...
  double correspondence_update_tolerance_rot;
  double correspondence_update_tolerance_trans;
  mutable Eigen::Isometry3d last_correspondence_point;
  mutable std::vector<long> correspondences;
  mutable std::vector<Eigen::Matrix4d> mahalanobis;

  std::shared_ptr<const TargetFrame> target;
  std::shared_ptr<const SourceFrame> source;
};

using IntegratedGICPFactor = IntegratedGICPFactor_<>;

}  // namespace gtsam_points