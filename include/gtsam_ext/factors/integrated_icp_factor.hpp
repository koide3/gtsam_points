// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>

#include <memory>
#include <gtsam_ext/types/frame.hpp>
#include <gtsam_ext/factors/integrated_matching_cost_factor.hpp>

namespace gtsam_ext {

struct NearestNeighborSearch;

/**
 * @brief Naive point-to-point ICP matching cost factor
 * @ref Zhang, "Iterative Point Matching for Registration of Free-Form Curve", IJCV1994
 */
template <typename TargetFrame = gtsam_ext::Frame, typename SourceFrame = gtsam_ext::Frame>
class IntegratedICPFactor_ : public gtsam_ext::IntegratedMatchingCostFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedICPFactor_<Frame>>;

  IntegratedICPFactor_(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source,
    const std::shared_ptr<NearestNeighborSearch>& target_tree,
    bool use_point_to_plane = false);

  IntegratedICPFactor_(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source,
    bool use_point_to_plane = false);

  IntegratedICPFactor_(
    const gtsam::Pose3& fixed_target_pose,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source,
    const std::shared_ptr<NearestNeighborSearch>& target_tree,
    bool use_point_to_plane = false);

  IntegratedICPFactor_(
    const gtsam::Pose3& fixed_target_pose,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source,
    bool use_point_to_plane = false);

  virtual ~IntegratedICPFactor_() override;

  // note: If your GTSAM is built with TBB, linearization is already multi-threaded
  //     : and setting n>1 can rather affect the processing speed
  void set_num_threads(int n) { num_threads = n; }
  void set_max_corresponding_distance(double dist) { max_correspondence_distance_sq = dist * dist; }
  void set_point_to_plane_distance(bool use) { use_point_to_plane = use; }
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

  std::shared_ptr<NearestNeighborSearch> target_tree;

  // I'm unhappy to have mutable members...
  double correspondence_update_tolerance_rot;
  double correspondence_update_tolerance_trans;
  mutable Eigen::Isometry3d last_correspondence_point;
  mutable std::vector<int> correspondences;

  std::shared_ptr<const TargetFrame> target;
  std::shared_ptr<const SourceFrame> source;
};

template <typename TargetFrame = gtsam_ext::Frame, typename SourceFrame = gtsam_ext::Frame>
class IntegratedPointToPlaneICPFactor_ : public gtsam_ext::IntegratedICPFactor_<TargetFrame, SourceFrame> {
public:
  using shared_ptr = boost::shared_ptr<IntegratedPointToPlaneICPFactor_<TargetFrame, SourceFrame>>;

  IntegratedPointToPlaneICPFactor_(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source)
  : IntegratedICPFactor_<TargetFrame, SourceFrame>(target_key, source_key, target, source, true) {}
};

using IntegratedICPFactor = IntegratedICPFactor_<>;
using IntegratedPointToPlaneICPFactor = IntegratedPointToPlaneICPFactor_<>;

}  // namespace gtsam_ext