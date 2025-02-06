// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>

#include <memory>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/factors/intensity_gradients.hpp>
#include <gtsam_points/factors/integrated_matching_cost_factor.hpp>

namespace gtsam_points {

struct NearestNeighborSearch;

/**
 * @brief Photometric consistency factor between point clouds
 *
 * @note  This factor uses (x, y, z, intensity) to query nearest neighbor search
 *        The 4th element (intensity) will be simply ignored if a standard gtsam_points::KdTree is given
 *        while it can provide additional distance information between points if gtsam_points::IntensityKdTree is used
 *
 * @note  While the use of IntensityKdTree significantly improves the convergence speed,
 *        it can affect optimization stability in some cases
 *
 *        Park et al., "Colored Point Cloud Registration Revisited", ICCV2017
 */
template <
  typename TargetFrame = gtsam_points::PointCloud,
  typename SourceFrame = gtsam_points::PointCloud,
  typename IntensityGradients = gtsam_points::IntensityGradients>
class IntegratedColorConsistencyFactor_ : public gtsam_points::IntegratedMatchingCostFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedColorConsistencyFactor_<TargetFrame, SourceFrame>>;

  /**
   * @brief Create a binary color consistency factor between target and source poses.
   * @param target_key          Target key
   * @param source_key          Source key
   * @param target              Target point cloud frame
   * @param source              Source point cloud frame
   * @param target_tree         Target nearest neighbor search
   * @param target_gradients    Target intensity gradients
   */
  IntegratedColorConsistencyFactor_(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source,
    const std::shared_ptr<const NearestNeighborSearch>& target_tree,
    const std::shared_ptr<const IntensityGradients>& target_gradients);

  /**
   * @brief Create a unary color consistency factor between a fixed target pose and an active source pose.
   * @param target_key          Target key
   * @param source_key          Source key
   * @param target              Target point cloud frame
   * @param source              Source point cloud frame
   * @param target_tree         Target nearest neighbor search
   * @param target_gradients    Target intensity gradients
   */
  IntegratedColorConsistencyFactor_(
    const gtsam::Pose3& fixed_target_pose,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source,
    const std::shared_ptr<const NearestNeighborSearch>& target_tree,
    const std::shared_ptr<const IntensityGradients>& target_gradients);

  virtual ~IntegratedColorConsistencyFactor_() override;

  /// @brief Print the factor information.
  virtual void print(const std::string& s = "", const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const override;

  /// @brief Set the number of thread used for linearization of this factor.
  /// @note If your GTSAM is built with TBB, linearization is already multi-threaded
  ///       and setting n>1 can rather affect the processing speed.
  void set_num_threads(int n) { num_threads = n; }

  /// @brief Set the maximum distance between corresponding points.
  ///        Correspondences with distances larger than this will be rejected (i.e., correspondence trimming).
  void set_max_correspondence_distance(double d) { max_correspondence_distance_sq = d * d; }

  /// @brief Set the weight for the photometric error term.
  void set_photometric_term_weight(double w) { photometric_term_weight = w; }

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
  double photometric_term_weight;  // [0, 1]

  std::shared_ptr<const TargetFrame> target;
  std::shared_ptr<const SourceFrame> source;
  std::shared_ptr<const NearestNeighborSearch> target_tree;
  std::shared_ptr<const IntensityGradients> target_gradients;

  double correspondence_update_tolerance_rot;
  double correspondence_update_tolerance_trans;
  mutable Eigen::Isometry3d last_correspondence_point;
  mutable std::vector<long> correspondences;
};

using IntegratedColorConsistencyFactor = IntegratedColorConsistencyFactor_<>;

}  // namespace gtsam_points