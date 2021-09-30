#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>

#include <memory>
#include <gtsam_ext/types/frame.hpp>

namespace gtsam_ext {

/**
 * @brief Abstraction of LSQ-based scan matching constraints between point clouds
 */
class IntegratedMatchingCostFactor : public gtsam::NonlinearFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedMatchingCostFactor>;

  IntegratedMatchingCostFactor(gtsam::Key target_key, gtsam::Key source_key);
  virtual ~IntegratedMatchingCostFactor() override;

  virtual size_t dim() const override { return 6; }

  // caution: The following error and linearize methods are not thread-safe,
  //        : because we need to update correspondences (that may be mutable members) for every linearization
  virtual double error(const gtsam::Values& values) const override;
  virtual boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& values) const override;

public:
  Eigen::Isometry3d calc_delta(const gtsam::Values& values) const;

  virtual void update_correspondences(const Eigen::Isometry3d& delta) const = 0;

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
}  // namespace gtsam_ext