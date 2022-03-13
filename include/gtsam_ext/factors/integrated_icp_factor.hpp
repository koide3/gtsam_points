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
template <typename Frame = gtsam_ext::Frame>
class IntegratedICPFactor : public gtsam_ext::IntegratedMatchingCostFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedICPFactor<Frame>>;

  IntegratedICPFactor(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const Frame>& target,
    const std::shared_ptr<const Frame>& source,
    const std::shared_ptr<NearestNeighborSearch>& target_tree,
    bool use_point_to_plane = false);

  IntegratedICPFactor(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const Frame>& target,
    const std::shared_ptr<const Frame>& source,
    bool use_point_to_plane = false);

  IntegratedICPFactor(
    const gtsam::Pose3& fixed_target_pose,
    gtsam::Key source_key,
    const std::shared_ptr<const Frame>& target,
    const std::shared_ptr<const Frame>& source,
    const std::shared_ptr<NearestNeighborSearch>& target_tree,
    bool use_point_to_plane = false);

  IntegratedICPFactor(
    const gtsam::Pose3& fixed_target_pose,
    gtsam::Key source_key,
    const std::shared_ptr<const Frame>& target,
    const std::shared_ptr<const Frame>& source,
    bool use_point_to_plane = false);

  virtual ~IntegratedICPFactor() override;

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

  std::shared_ptr<const Frame> target;
  std::shared_ptr<const Frame> source;
};

template <typename Frame = gtsam_ext::Frame>
class IntegratedPointToPlaneICPFactor : public gtsam_ext::IntegratedICPFactor<Frame> {
public:
  using shared_ptr = boost::shared_ptr<IntegratedPointToPlaneICPFactor<Frame>>;

  IntegratedPointToPlaneICPFactor(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const Frame>& target,
    const std::shared_ptr<const Frame>& source)
  : IntegratedICPFactor<Frame>(target_key, source_key, target, source, true) {}
};

}  // namespace gtsam_ext