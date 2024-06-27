/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    DoglegOptimizer.h
 * @brief
 * @author  Richard Roberts
 * @date   Feb 26, 2012
 */

#pragma once

#include <gtsam/nonlinear/NonlinearOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>

namespace gtsam_points {

/**
 * This class performs Dogleg nonlinear optimization
 */
class GTSAM_EXPORT DoglegOptimizerExt : public gtsam::NonlinearOptimizer {
protected:
  gtsam::DoglegParams params_;

public:
  typedef boost::shared_ptr<DoglegOptimizerExt> shared_ptr;

  /// @name Standard interface
  /// @{

  /** Standard constructor, requires a nonlinear factor graph, initial
   * variable assignments, and optimization parameters.  For convenience this
   * version takes plain objects instead of shared pointers, but internally
   * copies the objects.
   * @param graph The nonlinear factor graph to optimize
   * @param initialValues The initial variable assignments
   * @param params The optimization parameters
   */
  DoglegOptimizerExt(
    const gtsam::NonlinearFactorGraph& graph,
    const gtsam::Values& initialValues,
    const gtsam::DoglegParams& params = gtsam::DoglegParams());

  /** Standard constructor, requires a nonlinear factor graph, initial
   * variable assignments, and optimization parameters.  For convenience this
   * version takes plain objects instead of shared pointers, but internally
   * copies the objects.
   * @param graph The nonlinear factor graph to optimize
   * @param initialValues The initial variable assignments
   */
  DoglegOptimizerExt(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& initialValues, const gtsam::Ordering& ordering);

  /// @}

  /// @name Advanced interface
  /// @{

  /** Virtual destructor */
  ~DoglegOptimizerExt() override {}

  /**
   * Perform a single iteration, returning GaussianFactorGraph corresponding to
   * the linearized factor graph.
   */
  gtsam::GaussianFactorGraph::shared_ptr iterate() override;

  /** Read-only access the parameters */
  const gtsam::DoglegParams& params() const { return params_; }

  /** Access the current trust region radius delta */
  double getDelta() const;

  /// @}

protected:
  /** Access the parameters (base class version) */
  const gtsam::NonlinearOptimizerParams& _params() const override { return params_; }

  /** Internal function for computing a COLAMD ordering if no ordering is specified */
  gtsam::DoglegParams ensureHasOrdering(gtsam::DoglegParams params, const gtsam::NonlinearFactorGraph& graph) const;
};

}  // namespace gtsam_points
