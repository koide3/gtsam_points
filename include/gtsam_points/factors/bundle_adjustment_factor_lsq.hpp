// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam_points/factors/bundle_adjustment_factor.hpp>

namespace gtsam_points {

/**
 * @brief Bundle adjustment factor based on EVM and EF optimal condition satisfaction
 *
 * @note  The evaluation cost of this factor depends on the number of frames
 *        and is independent of the number of points
 * @note  This factor requires a better initial guess compared to EVM-based one
 *        because the global normal not included in the optimization
 *
 *        Huang et al, "On Bundle Adjustment for Multiview Point Cloud Registration", IEEE RA-L, 2021
 */
class LsqBundleAdjustmentFactor : public BundleAdjustmentFactorBase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<LsqBundleAdjustmentFactor>;

  LsqBundleAdjustmentFactor();
  virtual ~LsqBundleAdjustmentFactor() override;

  /// @brief Print the factor information.
  virtual void print(const std::string& s = "", const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const override;

  virtual size_t dim() const override { return 6; }
  virtual double error(const gtsam::Values& c) const override;
  virtual boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& c) const override;

  virtual void add(const gtsam::Point3& pt, const gtsam::Key& key) override;
  virtual int num_points() const override { return global_num_points; }

private:
  void update_global_distribution(const gtsam::Values& values) const;

protected:
  struct FrameDistribution;

  std::unordered_map<gtsam::Key, std::shared_ptr<FrameDistribution>> frame_dists;

  int global_num_points;
  mutable Eigen::Vector3d global_mean;
  mutable Eigen::Matrix3d global_cov;
  mutable Eigen::Vector3d global_normal;
};

}  // namespace gtsam_points
