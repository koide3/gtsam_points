// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>

#include <memory>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/factors/integrated_matching_cost_factor.hpp>

namespace gtsam_points {

struct NearestNeighborSearch;

template <typename TargetFrame, typename SourceFrame>
class IntegratedPointToPlaneFactor_;
template <typename TargetFrame, typename SourceFrame>
class IntegratedPointToEdgeFactor_;

/**
 * @brief Scan matching factor based on the combination of point-to-plane and point-to-edge distances
 *
 * Zhang and Singh, "Low-drift and real-time lidar odometry and mapping", Autonomous Robots, 2017
 * Zhang and Singh, "LOAM: LiDAR Odometry and Mapping in Real-time", RSS2014
 * Tixiao and Brendan, "LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain", IROS2018
 */
template <typename TargetFrame = gtsam_points::PointCloud, typename SourceFrame = gtsam_points::PointCloud>
class IntegratedLOAMFactor_ : public gtsam_points::IntegratedMatchingCostFactor {
public:
  GTSAM_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedLOAMFactor_<TargetFrame, SourceFrame>>;

  IntegratedLOAMFactor_(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target_edges,
    const std::shared_ptr<const TargetFrame>& target_planes,
    const std::shared_ptr<const SourceFrame>& source_edges,
    const std::shared_ptr<const SourceFrame>& source_planes,
    const std::shared_ptr<const NearestNeighborSearch>& target_edges_tree,
    const std::shared_ptr<const NearestNeighborSearch>& target_planes_tree);

  IntegratedLOAMFactor_(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target_edges,
    const std::shared_ptr<const TargetFrame>& target_planes,
    const std::shared_ptr<const SourceFrame>& source_edges,
    const std::shared_ptr<const SourceFrame>& source_planes);

  ~IntegratedLOAMFactor_();

  virtual void print(const std::string& s = "", const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const override;

  // note: If your GTSAM is built with TBB, linearization is already multi-threaded
  //     : and setting n>1 can rather affect the processing speed
  void set_num_threads(int n);
  void set_max_correspondence_distance(double dist_edge, double dist_plane);
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
  std::unique_ptr<IntegratedPointToEdgeFactor_<TargetFrame, SourceFrame>> edge_factor;
  std::unique_ptr<IntegratedPointToPlaneFactor_<TargetFrame, SourceFrame>> plane_factor;
};

// Point-to-plane distance
template <typename TargetFrame = gtsam_points::PointCloud, typename SourceFrame = gtsam_points::PointCloud>
class IntegratedPointToPlaneFactor_ : public gtsam_points::IntegratedMatchingCostFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedPointToPlaneFactor_<TargetFrame, SourceFrame>>;

  friend class IntegratedLOAMFactor_<TargetFrame, SourceFrame>;

  IntegratedPointToPlaneFactor_(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source,
    const std::shared_ptr<const NearestNeighborSearch>& target_tree);

  IntegratedPointToPlaneFactor_(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source);

  ~IntegratedPointToPlaneFactor_();

  virtual void print(const std::string& s = "", const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const override;

  void set_num_threads(int n) { num_threads = n; }
  void set_max_correspondence_distance(double dist) { max_correspondence_distance_sq = dist * dist; }
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

  std::shared_ptr<const NearestNeighborSearch> target_tree;

  // I'm unhappy to have mutable members...
  double correspondence_update_tolerance_rot;
  double correspondence_update_tolerance_trans;
  mutable Eigen::Isometry3d last_correspondence_point;
  mutable std::vector<std::tuple<long, long, long>> correspondences;

  std::shared_ptr<const TargetFrame> target;
  std::shared_ptr<const SourceFrame> source;
};

// Point-to-edge distance
template <typename TargetFrame = gtsam_points::PointCloud, typename SourceFrame = gtsam_points::PointCloud>
class IntegratedPointToEdgeFactor_ : public gtsam_points::IntegratedMatchingCostFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedPointToEdgeFactor_<TargetFrame, SourceFrame>>;

  friend class IntegratedLOAMFactor_<TargetFrame, SourceFrame>;

  IntegratedPointToEdgeFactor_(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source,
    const std::shared_ptr<const NearestNeighborSearch>& target_tree);

  IntegratedPointToEdgeFactor_(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source);

  ~IntegratedPointToEdgeFactor_();

  virtual void print(const std::string& s = "", const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const override;

  void set_num_threads(int n) { num_threads = n; }
  void set_max_correspondence_distance(double dist) { max_correspondence_distance_sq = dist * dist; }
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

  std::shared_ptr<const NearestNeighborSearch> target_tree;

  // I'm unhappy to have mutable members...
  double correspondence_update_tolerance_rot;
  double correspondence_update_tolerance_trans;
  mutable Eigen::Isometry3d last_correspondence_point;
  mutable std::vector<std::tuple<long, long>> correspondences;

  std::shared_ptr<const TargetFrame> target;
  std::shared_ptr<const SourceFrame> source;
};

using IntegratedLOAMFactor = gtsam_points::IntegratedLOAMFactor_<>;
using IntegratedPointToPlaneFactor = gtsam_points::IntegratedPointToPlaneFactor_<>;
using IntegratedPointToEdgeFactor = gtsam_points::IntegratedPointToEdgeFactor_<>;

}  // namespace gtsam_points
