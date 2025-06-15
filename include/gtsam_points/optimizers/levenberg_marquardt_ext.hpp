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

#include <gtsam/nonlinear/NonlinearOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/linear/VectorValues.h>
#include <chrono>

#include <gtsam_points/optimizers/levenberg_marquardt_ext_params.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_optimization_status.hpp>

namespace gtsam_points {

class LinearizationHook;

/**
 * This class performs Levenberg-Marquardt nonlinear optimization
 */
class GTSAM_EXPORT LevenbergMarquardtOptimizerExt : public gtsam::NonlinearOptimizer {
protected:
  const LevenbergMarquardtExtParams params_;  ///< LM parameters
  const std::unique_ptr<LinearizationHook> linearization_hook_;

  gtsam::GaussianFactorGraph::shared_ptr linearized_;

  // startTime_ is a chrono time point
  std::chrono::time_point<std::chrono::high_resolution_clock> startTime_;  ///< time when optimization started
  std::chrono::high_resolution_clock::time_point optimization_start_time_;
  std::chrono::nanoseconds linearization_time_;

  void initTime();

public:
  typedef std::shared_ptr<LevenbergMarquardtOptimizerExt> shared_ptr;

  /// @name Constructors/Destructor
  /// @{

  /** Standard constructor, requires a nonlinear factor graph, initial
   * variable assignments, and optimization parameters.  For convenience this
   * version takes plain objects instead of shared pointers, but internally
   * copies the objects.
   * @param graph The nonlinear factor graph to optimize
   * @param initialValues The initial variable assignments
   * @param params The optimization parameters
   */
  LevenbergMarquardtOptimizerExt(
    const gtsam::NonlinearFactorGraph& graph,
    const gtsam::Values& initialValues,
    const LevenbergMarquardtExtParams& params = LevenbergMarquardtExtParams());

  /** Standard constructor, requires a nonlinear factor graph, initial
   * variable assignments, and optimization parameters.  For convenience this
   * version takes plain objects instead of shared pointers, but internally
   * copies the objects.
   * @param graph The nonlinear factor graph to optimize
   * @param initialValues The initial variable assignments
   */
  LevenbergMarquardtOptimizerExt(
    const gtsam::NonlinearFactorGraph& graph,
    const gtsam::Values& initialValues,
    const gtsam::Ordering& ordering,
    const LevenbergMarquardtExtParams& params = LevenbergMarquardtExtParams());

  /** Virtual destructor */
  ~LevenbergMarquardtOptimizerExt() override;

  /// @}

  /// @name Standard interface
  /// @{

  /// Access the current damping value
  double lambda() const;

  /// Access the current number of inner iterations
  int getInnerIterations() const;

  /// print
  void print(const std::string& str = "") const {
    std::cout << str << "LevenbergMarquardtOptimizerExt" << std::endl;
    this->params_.print("  parameters:\n");
  }

  /// @}

  /// @name Advanced interface
  /// @{

  /**
   * Perform a single iteration, returning GaussianFactorGraph corresponding to
   * the linearized factor graph.
   */
  gtsam::GaussianFactorGraph::shared_ptr iterate() override;

  /** Read-only access the parameters */
  const LevenbergMarquardtExtParams& params() const { return params_; }

  void writeLogFile(double currentError);

  /** linearize, can be overwritten */
  virtual gtsam::GaussianFactorGraph::shared_ptr linearize() const;

  /** Build a damped system for a specific lambda -- for testing only */
  gtsam::GaussianFactorGraph buildDampedSystem(const gtsam::GaussianFactorGraph& linear, const gtsam::VectorValues& sqrtHessianDiagonal) const;

  /** Inner loop, changes state, returns true if successful or giving up */
  bool tryLambda(const gtsam::GaussianFactorGraph& linear, const gtsam::VectorValues& sqrtHessianDiagonal);

  const gtsam::Values& optimize() override;

  /// @}

protected:
  /** Access the parameters (base class version) */
  const gtsam::NonlinearOptimizerParams& _params() const override { return params_; }
};

}  // namespace gtsam_points