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
 * @brief Colored GICP matching cost factor
 *
 * @note  This factor uses (x, y, z, intensity) to query nearest neighbor search
 *        The 4th element (intensity) will be simply ignored if a standard gtsam_points::KdTree is given
 *        while it can provide additional distance information between points if gtsam_points::IntensityKdTree is used
 *
 * @note  While the use of IntensityKdTree significantly improves the convergence speed,
 *        it can affect optimization stability in some cases
 *
 *        Segal et al., "Generalized-ICP", RSS2005
 *        Park et al., "Colored Point Cloud Registration Revisited", ICCV2017
 */
template <
  typename TargetFrame = gtsam_points::PointCloud,
  typename SourceFrame = gtsam_points::PointCloud,
  typename IntensityGradients = gtsam_points::IntensityGradients>
class IntegratedColoredGICPFactor_ : public gtsam_points::IntegratedMatchingCostFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedColoredGICPFactor_<TargetFrame, SourceFrame>>;

  IntegratedColoredGICPFactor_(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source,
    const std::shared_ptr<const NearestNeighborSearch>& target_tree,
    const std::shared_ptr<const IntensityGradients>& target_gradients);

  IntegratedColoredGICPFactor_(
    const gtsam::Pose3& fixed_target_pose,
    gtsam::Key source_key,
    const std::shared_ptr<const TargetFrame>& target,
    const std::shared_ptr<const SourceFrame>& source,
    const std::shared_ptr<const NearestNeighborSearch>& target_tree,
    const std::shared_ptr<const IntensityGradients>& target_gradients);

  virtual ~IntegratedColoredGICPFactor_() override;

  virtual void print(const std::string& s = "", const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const override;

  void set_num_threads(int n) { num_threads = n; }
  void set_max_correspondence_distance(double d) { max_correspondence_distance_sq = d * d; }
  void set_photometric_term_weight(double w) { photometric_term_weight = w; }

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
  mutable std::vector<Eigen::Matrix4d> mahalanobis;
};

using IntegratedColoredGICPFactor = IntegratedColoredGICPFactor_<>;

}  // namespace gtsam_points