// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>

#include <memory>
#include <gtsam_ext/types/basic_frame.hpp>
#include <gtsam_ext/factors/integrated_matching_cost_factor.hpp>

namespace gtsam_ext {

struct NearestNeighborSearch;

template <typename Frame>
class IntegratedPointToPlaneFactor;
template <typename Frame>
class IntegratedPointToEdgeFactor;

/**
 * @brief Scan matching factor based on the combination of point-to-plane and point-to-edge distances
 *
 * @ref Zhang and Singh, "Low-drift and real-time lidar odometry and mapping", Autonomous Robots, 2017
 * @ref Zhang and Singh, "LOAM: LiDAR Odometry and Mapping in Real-time", RSS2014
 * @ref Tixiao and Brendan, "LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain", IROS2018
 */
template <typename Frame = BasicFrame>
class IntegratedLOAMFactor : public gtsam_ext::IntegratedMatchingCostFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedLOAMFactor>;

  IntegratedLOAMFactor(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const Frame>& target_edges,
    const std::shared_ptr<const Frame>& target_planes,
    const std::shared_ptr<const Frame>& source_edges,
    const std::shared_ptr<const Frame>& source_planes,
    const std::shared_ptr<NearestNeighborSearch>& target_edges_tree,
    const std::shared_ptr<NearestNeighborSearch>& target_planes_tree);

  IntegratedLOAMFactor(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const Frame>& target_edges,
    const std::shared_ptr<const Frame>& target_planes,
    const std::shared_ptr<const Frame>& source_edges,
    const std::shared_ptr<const Frame>& source_planes);

  ~IntegratedLOAMFactor();

  // note: If your GTSAM is built with TBB, linearization is already multi-threaded
  //     : and setting n>1 can rather affect the processing speed
  void set_num_threads(int n);
  void set_max_corresponding_distance(double dist_edge, double dist_plane);
  void set_correspondence_update_tolerance(double angle, double trans);
  void set_enable_correspondence_validation(bool enable);

private:
  virtual void update_correspondences(const Eigen::Isometry3d& delta) const override;

  virtual double evaluate(
    const Eigen::Isometry3d& delta,
    Eigen::Matrix<double, 6, 6>* H_target = nullptr,
    Eigen::Matrix<double, 6, 6>* H_source = nullptr,
    Eigen::Matrix<double, 6, 6>* H_target_source = nullptr,
    Eigen::Matrix<double, 6, 1>* b_target = nullptr,
    Eigen::Matrix<double, 6, 1>* b_source = nullptr) const override;

protected:
  // This method is called after update_correspondences()
  // You can override this to reject invalid correspondences
  // (e.g., rejecting edge correspondences in a same scan line)
  virtual void validate_correspondences() const;

private:
  bool enable_correspondence_validation;
  std::unique_ptr<IntegratedPointToEdgeFactor<Frame>> edge_factor;
  std::unique_ptr<IntegratedPointToPlaneFactor<Frame>> plane_factor;
};

// Point-to-plane distance
template <typename Frame = BasicFrame>
class IntegratedPointToPlaneFactor : public gtsam_ext::IntegratedMatchingCostFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedPointToPlaneFactor<Frame>>;

  friend class IntegratedLOAMFactor<Frame>;

  IntegratedPointToPlaneFactor(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const Frame>& target,
    const std::shared_ptr<const Frame>& source,
    const std::shared_ptr<NearestNeighborSearch>& target_tree);

  IntegratedPointToPlaneFactor(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const Frame>& target,
    const std::shared_ptr<const Frame>& source);

  ~IntegratedPointToPlaneFactor();

  void set_num_threads(int n) { num_threads = n; }
  void set_max_corresponding_distance(double dist) { max_correspondence_distance_sq = dist * dist; }
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

  std::shared_ptr<NearestNeighborSearch> target_tree;

  // I'm unhappy to have mutable members...
  double correspondence_update_tolerance_rot;
  double correspondence_update_tolerance_trans;
  mutable Eigen::Isometry3d last_correspondence_point;
  mutable std::vector<std::tuple<int, int, int>> correspondences;

  std::shared_ptr<const Frame> target;
  std::shared_ptr<const Frame> source;
};

// Point-to-edge distance
template <typename Frame = BasicFrame>
class IntegratedPointToEdgeFactor : public gtsam_ext::IntegratedMatchingCostFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedPointToEdgeFactor<Frame>>;

  friend class IntegratedLOAMFactor<Frame>;

  IntegratedPointToEdgeFactor(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const Frame>& target,
    const std::shared_ptr<const Frame>& source,
    const std::shared_ptr<NearestNeighborSearch>& target_tree);

  IntegratedPointToEdgeFactor(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const Frame>& target,
    const std::shared_ptr<const Frame>& source);

  ~IntegratedPointToEdgeFactor();

  void set_num_threads(int n) { num_threads = n; }
  void set_max_corresponding_distance(double dist) { max_correspondence_distance_sq = dist * dist; }
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

  std::shared_ptr<NearestNeighborSearch> target_tree;

  // I'm unhappy to have mutable members...
  double correspondence_update_tolerance_rot;
  double correspondence_update_tolerance_trans;
  mutable Eigen::Isometry3d last_correspondence_point;
  mutable std::vector<std::tuple<int, int>> correspondences;

  std::shared_ptr<const Frame> target;
  std::shared_ptr<const Frame> source;
};

}  // namespace gtsam_ext
