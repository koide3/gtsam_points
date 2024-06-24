/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    LevenbergMarquardtOptimizer.h
 * @brief   A nonlinear optimizer that uses the Levenberg-Marquardt trust-region scheme
 * @author  Richard Roberts
 * @author  Frank Dellaert
 * @author  Luca Carlone
 * @date    Feb 26, 2012
 */

#pragma once
#include <chrono>
#include <functional>
#include <boost/utility/in_place_factory.hpp>
#include <boost/utility/typed_in_place_factory.hpp>
#include <gtsam/linear/VectorValues.h>
#include <gtsam/nonlinear/NonlinearOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>

#include <gtsam_points/optimizers/levenberg_marquardt_ext_params.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_optimization_status.hpp>

namespace gtsam_points {

class LinearizationHook;

class LevenbergMarquardtOptimizerExt : public gtsam::NonlinearOptimizer {
public:
  LevenbergMarquardtOptimizerExt(
    const gtsam::NonlinearFactorGraph& graph,
    const gtsam::Values& initialValues,
    const LevenbergMarquardtExtParams& params = LevenbergMarquardtExtParams());

  ~LevenbergMarquardtOptimizerExt();

  virtual gtsam::GaussianFactorGraph::shared_ptr iterate() override;

  virtual const gtsam::Values& optimize() override;

  const gtsam::GaussianFactorGraph::shared_ptr& last_linearized() const { return linearized; }

private:
  gtsam::GaussianFactorGraph buildDampedSystem(const gtsam::GaussianFactorGraph& linear, const gtsam::VectorValues& sqrtHessianDiagonal) const;
  bool tryLambda(const gtsam::GaussianFactorGraph& linear, const gtsam::VectorValues& sqrtHessianDiagonal, double error);

  const gtsam::NonlinearOptimizerParams& _params() const override { return params_; }

private:
  std::chrono::high_resolution_clock::time_point optimization_start_time;
  std::chrono::nanoseconds linearization_time;

  std::unique_ptr<LinearizationHook> linearization_hook;
  const LevenbergMarquardtExtParams params_;

  gtsam::GaussianFactorGraph::shared_ptr linearized;
};
}  // namespace gtsam_points
