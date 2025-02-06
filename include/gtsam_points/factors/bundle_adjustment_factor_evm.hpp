// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <unordered_map>
#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam_points/factors/bundle_adjustment_factor.hpp>

namespace gtsam_points {

struct BALMFeature;

/**
 * @brief Bundle adjustment factor based on Eigenvalue minimization
 *        One EVMFactor represents one feature (plane / edge)
 *
 * @note  The computation cost grows as the number of points increases
 *        Consider averaging points in a same scan in advance (see [Liu 2021])
 *
 *        Liu and Zhang, "BALM: Bundle Adjustment for Lidar Mapping", IEEE RA-L, 2021
 */
class EVMBundleAdjustmentFactorBase : public BundleAdjustmentFactorBase {
public:
  using shared_ptr = boost::shared_ptr<EVMBundleAdjustmentFactorBase>;

  EVMBundleAdjustmentFactorBase();
  virtual ~EVMBundleAdjustmentFactorBase() override;
  virtual size_t dim() const override { return 6; }

  /// @brief Print the factor information.
  virtual void print(const std::string& s = "", const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const override;

  /**
   * @brief  Assign a point to the factor
   * @param  pt       Point to be added
   * @param  key      Key of the pose corresponding to the point
   */
  virtual void add(const gtsam::Point3& pt, const gtsam::Key& key) override;

  /**
   * @brief Number of points assigned to this factor
   * @return Number of points
   */
  virtual int num_points() const override { return points.size(); }

  /**
   * @brief  Set a constant error scaling factor to boost the weight of the factor
   * @param  scale  Error scale
   */
  virtual void set_scale(double scale) override;

protected:
  template <int k>
  double calc_eigenvalue(const std::vector<Eigen::Vector3d>& transed_points, Eigen::MatrixXd* H = nullptr, Eigen::MatrixXd* J = nullptr) const;

  Eigen::MatrixXd calc_pose_derivatives(const std::vector<Eigen::Vector3d>& transed_points) const;

  gtsam::GaussianFactor::shared_ptr compose_factor(const Eigen::MatrixXd& H, const Eigen::MatrixXd& J, double error) const;

protected:
  double error_scale;
  std::vector<gtsam::Key> keys;
  std::vector<gtsam::Point3> points;
  std::unordered_map<gtsam::Key, int> key_index;
};

/**
 * @brief Plane EVM factor that minimizes lambda_0
 */
class PlaneEVMFactor : public EVMBundleAdjustmentFactorBase {
public:
  using shared_ptr = boost::shared_ptr<PlaneEVMFactor>;

  PlaneEVMFactor();
  virtual ~PlaneEVMFactor() override;

  /// @brief Print the factor information.
  virtual void print(const std::string& s = "", const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const override;

  virtual double error(const gtsam::Values& c) const override;
  virtual boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& c) const override;

  // TODO: Add feature parameter extraction method
};

/**
 * @brief Edge EVM factor that minimizes lambda_0 + lambda_1
 */
class EdgeEVMFactor : public EVMBundleAdjustmentFactorBase {
public:
  using shared_ptr = boost::shared_ptr<EdgeEVMFactor>;

  EdgeEVMFactor();
  virtual ~EdgeEVMFactor() override;

  /// @brief Print the factor information.
  virtual void print(const std::string& s = "", const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const override;

  virtual double error(const gtsam::Values& c) const override;
  virtual boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& c) const override;
};
}  // namespace gtsam_points