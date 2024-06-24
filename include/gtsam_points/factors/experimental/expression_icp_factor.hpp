// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam/slam/expressions.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam_points/types/point_cloud.hpp>

namespace gtsam_points {

class KdTree;

/**
 * @brief ICP distance factor with GTSAM's expression capability
 *        This is experimental and may not be suitable for practical use
 */
class ICPFactorExpr : public gtsam::NoiseModelFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ICPFactorExpr(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const std::shared_ptr<const PointCloud>& target,
    const std::shared_ptr<const KdTree>& target_tree,
    const gtsam::Point3& source,
    const gtsam::SharedNoiseModel& noise_model);

  ~ICPFactorExpr();

  virtual gtsam::Vector unwhitenedError(const gtsam::Values& values, boost::optional<std::vector<gtsam::Matrix>&> H = boost::none) const;

  void update_correspondence(const gtsam::Values& values) const;

  gtsam::Point3_ calc_error() const;

private:
  const std::shared_ptr<const PointCloud> target;
  const std::shared_ptr<const KdTree> target_tree;

  const gtsam::Pose3_ delta;
  const gtsam::Point3 source;

  mutable int target_index;
  mutable gtsam::Point3_ error_expr;
};

/**
 * @brief Create a set of ICPFactorExpr
 */
gtsam::NonlinearFactorGraph::shared_ptr
create_icp_factors(gtsam::Key target_key, gtsam::Key source_key, const PointCloud::ConstPtr& target, const PointCloud::ConstPtr& source, const gtsam::SharedNoiseModel& noise_model);

/**
 * @brief Create a nonlinear factor that wraps a set of ICP factors
 */
gtsam::NonlinearFactor::shared_ptr create_integrated_icp_factor(
  gtsam::Key target_key,
  gtsam::Key source_key,
  const PointCloud::ConstPtr& target,
  const PointCloud::ConstPtr& source,
  const gtsam::SharedNoiseModel& noise_model);

}  // namespace gtsam_points