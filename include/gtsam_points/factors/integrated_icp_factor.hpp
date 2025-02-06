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
 * @brief Naive point-to-point ICP matching cost factor
 *        Zhang, "Iterative Point Matching for Registration of Free-Form Curve", IJCV1994
 */
template <typename TargetFrame = gtsam_points::PointCloud, typename SourceFrame = gtsam_points::PointCloud>
class IntegratedICPFactor_ : public gtsam_points::IntegratedMatchingCostFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedICPFactor_<PointCloud>>;

  /**
   * @brief Create a binary ICP factor between two poses.
   * @param target_key          Target key
   * @param source_key          Source key
   * @param target              Target point cloud frame
   * @param source              Source point cloud frame
   * @param target_tree         Target nearest neighbor search
   * @param use_point_to_plane  If true, use point-to-plane distance instead of point-to-point distance
   */
  IntegratedICPFactor_(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source,
    const std::shared_ptr<const NearestNeighborSearch>& target_tree,
    bool use_point_to_plane = false);

  /// Create a binary ICP factor between two poses.
  IntegratedICPFactor_(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source,
    bool use_point_to_plane = false);

  /**
   * @brief Create a unary ICP factor between a fixed target pose and an active source pose.
   * @param fixed_target_pose   Fixed target pose
   * @param source_key          Source key
   * @param target              Target point cloud frame
   * @param source              Source point cloud frame
   * @param target_tree         Target nearest neighbor search
   * @param use_point_to_plane  If true, use point-to-plane distance instead of point-to-point distance
   */
  IntegratedICPFactor_(
    const gtsam::Pose3& fixed_target_pose,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source,
    const std::shared_ptr<const NearestNeighborSearch>& target_tree,
    bool use_point_to_plane = false);

  /// Create a unary ICP factor between a fixed target pose and an active source pose.
  IntegratedICPFactor_(
    const gtsam::Pose3& fixed_target_pose,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source,
    bool use_point_to_plane = false);

  virtual ~IntegratedICPFactor_() override;

  /// @brief Print the factor information.
  virtual void print(const std::string& s = "", const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const override;

  /// @brief Set the number of thread used for linearization of this factor.
  /// @note If your GTSAM is built with TBB, linearization is already multi-threaded
  ///       and setting n>1 can rather affect the processing speed.
  void set_num_threads(int n) { num_threads = n; }

  /// @brief Set the maximum distance between corresponding points.
  ///        Correspondences with distances larger than this will be rejected (i.e., correspondence trimming).
  void set_max_correspondence_distance(double dist) { max_correspondence_distance_sq = dist * dist; }

  /// @brief Enable or disable point-to-plane distance computation.
  void set_point_to_plane_distance(bool use) { use_point_to_plane = use; }

  /// @brief Correspondences are updated only when the displacement from the last update point is larger than these threshold values.
  /// @note  Default values are angle=trans=0 and correspondences are updated every linearization call.
  void set_correspondence_update_tolerance(double angle, double trans) {
    correspondence_update_tolerance_rot = angle;
    correspondence_update_tolerance_trans = trans;
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
  bool use_point_to_plane;

  std::shared_ptr<const NearestNeighborSearch> target_tree;

  // I'm unhappy to have mutable members...
  double correspondence_update_tolerance_rot;
  double correspondence_update_tolerance_trans;
  mutable Eigen::Isometry3d last_correspondence_point;
  mutable std::vector<long> correspondences;

  std::shared_ptr<const TargetFrame> target;
  std::shared_ptr<const SourceFrame> source;
};

/**
 * @brief Point-to-plane ICP factor
 */
template <typename TargetFrame = gtsam_points::PointCloud, typename SourceFrame = gtsam_points::PointCloud>
class IntegratedPointToPlaneICPFactor_ : public gtsam_points::IntegratedICPFactor_<TargetFrame, SourceFrame> {
public:
  using shared_ptr = boost::shared_ptr<IntegratedPointToPlaneICPFactor_<TargetFrame, SourceFrame>>;

  IntegratedPointToPlaneICPFactor_(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source,
    const std::shared_ptr<const NearestNeighborSearch>& target_tree)
  : IntegratedICPFactor_<TargetFrame, SourceFrame>(target_key, source_key, target, source, target_tree, true) {}

  IntegratedPointToPlaneICPFactor_(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source)
  : IntegratedICPFactor_<TargetFrame, SourceFrame>(target_key, source_key, target, source, true) {}
};

using IntegratedICPFactor = IntegratedICPFactor_<>;
using IntegratedPointToPlaneICPFactor = IntegratedPointToPlaneICPFactor_<>;

}  // namespace gtsam_points