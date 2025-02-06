// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>

#include <memory>
#include <gtsam/inference/Key.h>
#include <gtsam/geometry/Pose3.h>

namespace gtsam_points {

/**
 * @brief Abstraction of LSQ-based scan matching constraints between point clouds
 */
class IntegratedMatchingCostFactor : public gtsam::NonlinearFactor {
public:
  GTSAM_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedMatchingCostFactor>;

  /**
   * @brief Create a binary matching cost factor between target and source poses
   * @param target_key  Target key
   * @param source_key  Source key
   */
  IntegratedMatchingCostFactor(gtsam::Key target_key, gtsam::Key source_key);

  /**
   * @brief Create a unary matching cost factor between a fixed target pose and an active source pose
   * @param fixed_target_pose  Fixed target pose
   * @param source_key         Source key
   */
  IntegratedMatchingCostFactor(const gtsam::Pose3& fixed_target_pose, gtsam::Key source_key);

  virtual ~IntegratedMatchingCostFactor() override;

  virtual size_t dim() const override { return 6; }

  /// @brief Print the factor information.
  virtual void print(const std::string& s = "", const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const override;

  /// @note The following error and linearize methods are not thread-safe,
  ///       because we need to update correspondences (that may be mutable members) for every linearization
  virtual double error(const gtsam::Values& values) const override;
  virtual boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& values) const override;

  const Eigen::Isometry3d& get_fixed_target_pose() const { return fixed_target_pose; }

public:
  Eigen::Isometry3d calc_delta(const gtsam::Values& values) const;

  /**
   * @brief Update point correspondences
   * @param delta Transformation between target and source (i.e., T_target_source)
   */
  virtual void update_correspondences(const Eigen::Isometry3d& delta) const = 0;

  /**
   * @brief Evaluate the matching cost
   * @param delta Transformation between target and source (i.e., T_target_source)
   * @param H_target         Hessian (target x target)
   * @param H_source         Hessian (source x source)
   * @param H_target_source  Hessian (target x source)
   * @param b_target         Error vector (target)
   * @param b_source         Error vector (source)
   */
  virtual double evaluate(
    const Eigen::Isometry3d& delta,
    Eigen::Matrix<double, 6, 6>* H_target = nullptr,
    Eigen::Matrix<double, 6, 6>* H_source = nullptr,
    Eigen::Matrix<double, 6, 6>* H_target_source = nullptr,
    Eigen::Matrix<double, 6, 1>* b_target = nullptr,
    Eigen::Matrix<double, 6, 1>* b_source = nullptr) const = 0;

protected:
  bool is_binary;
  Eigen::Isometry3d fixed_target_pose;
};
}  // namespace gtsam_points