/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    LevenbergMarquardtOptimizer.cpp
 * @brief   A nonlinear optimizer that uses the Levenberg-Marquardt trust-region scheme
 * @author  Richard Roberts
 * @author  Frank Dellaert
 * @author  Luca Carlone
 * @date    Feb 26, 2012
 */

#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>

#include <gtsam_points/optimizers/linearization_hook.hpp>

#include <gtsam/base/Vector.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/internal/LevenbergMarquardtState.h>
#include <gtsam/inference/Ordering.h>

#include <boost/format.hpp>
#include <boost/range/adaptor/map.hpp>

namespace gtsam_points {

using State = gtsam::internal::LevenbergMarquardtState;

LevenbergMarquardtExtParams LevenbergMarquardtExtParams::ensureHasOrdering(const gtsam::NonlinearFactorGraph& graph) const {
  if (ordering) {
    return *this;
  }

  LevenbergMarquardtExtParams params = *this;
  params.ordering = gtsam::Ordering::Create(orderingType, graph);
  return params;
}

LevenbergMarquardtOptimizerExt::LevenbergMarquardtOptimizerExt(
  const gtsam::NonlinearFactorGraph& graph,
  const gtsam::Values& initialValues,
  const LevenbergMarquardtExtParams& params)
: gtsam::NonlinearOptimizer(
    graph,
    std::unique_ptr<State>(new State(initialValues, std::numeric_limits<double>::max(), params.lambdaInitial, params.lambdaFactor))),
  params_(params.ensureHasOrdering(graph)) {
  // find gpu factors
  linearization_hook.reset(new LinearizationHook());
  linearization_hook->add(graph);
}

LevenbergMarquardtOptimizerExt::~LevenbergMarquardtOptimizerExt() {}

gtsam::GaussianFactorGraph LevenbergMarquardtOptimizerExt::buildDampedSystem(
  const gtsam::GaussianFactorGraph& linear,
  const gtsam::VectorValues& sqrtHessianDiagonal) const {
  auto currentState = static_cast<const State*>(state_.get());
  if (params_.diagonalDamping) {
    return currentState->buildDampedSystem(linear, sqrtHessianDiagonal);
  } else {
    return currentState->buildDampedSystem(linear);
  }
}

bool LevenbergMarquardtOptimizerExt::tryLambda(
  const gtsam::GaussianFactorGraph& linear,
  const gtsam::VectorValues& sqrtHessianDiagonal,
  double oldError) {
  auto lambda_iteration_start_time = std::chrono::high_resolution_clock::now();

  auto currentState = static_cast<const State*>(state_.get());
  auto dampedSystem = buildDampedSystem(linear, sqrtHessianDiagonal);

  // Try solving
  double modelFidelity = 0.0;
  bool step_is_successful = false;
  bool stopSearchingLambda = false;
  double newError = std::numeric_limits<double>::infinity(), costChange = 0.0;
  gtsam::Values newValues;
  gtsam::VectorValues delta;

  auto linear_solver_start_time = std::chrono::high_resolution_clock::now();
  bool systemSolvedSuccessfully;
  try {
    // ============ Solve is where most computation happens !! =================
    if (params_.solver) {
      delta = params_.solver->solve(dampedSystem, *params_.ordering);
    } else {
      delta = solve(dampedSystem, params_);
    }

    systemSolvedSuccessfully = true;
  } catch (const gtsam::IndeterminantLinearSystemException&) {
    systemSolvedSuccessfully = false;
  }
  auto linear_solver_end_time = std::chrono::high_resolution_clock::now();

  if (systemSolvedSuccessfully) {
    // Compute the old linearized error as it is not the same
    // as the nonlinear error when robust noise models are used.
    double oldLinearizedError = linear.error(gtsam::VectorValues::Zero(delta));
    double newlinearizedError = linear.error(delta);
    // cost change in the linearized system (old - new)
    double linearizedCostChange = oldLinearizedError - newlinearizedError;

    if (linearizedCostChange >= 0) {  // step is valid
      // ============ This is where the solution is updated ====================
      newValues = currentState->values.retract(delta);
      // =======================================================================

      // cost change in the original, nonlinear system (old - new)
      linearization_hook->error(newValues);
      newError = graph_.error(newValues);
      costChange = oldError - newError;

      if (linearizedCostChange > std::numeric_limits<double>::epsilon() * oldLinearizedError) {
        // the (linear) error has to decrease to satisfy this condition
        // fidelity of linearized model VS original system between
        modelFidelity = costChange / linearizedCostChange;
        // if we decrease the error in the nonlinear system and modelFidelity is above threshold
        step_is_successful = modelFidelity > params_.minModelFidelity;
      }  // else we consider the step non successful and we either increase lambda or stop if error
         // change is small

      double minAbsoluteTolerance = params_.relativeErrorTol * oldError;
      // if the change is small we terminate
      if (std::abs(costChange) < minAbsoluteTolerance) {
        stopSearchingLambda = true;
      }
    }
  }  // if (systemSolvedSuccessfully)

  if (params_.callback || params_.status_msg_callback) {
    LevenbergMarquardtOptimizationStatus status;
    status.iterations = currentState->iterations;
    status.total_inner_iterations = currentState->totalNumberInnerIterations;
    status.error = oldError;
    status.cost_change = costChange;
    status.lambda = currentState->lambda;
    status.solve_success = systemSolvedSuccessfully;

    auto t = std::chrono::high_resolution_clock::now();
    status.elapsed_time = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(t - optimization_start_time).count();
    status.linearization_time = 1e-9 * linearization_time.count();
    status.lambda_iteration_time = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(t - lambda_iteration_start_time).count();
    status.linear_solver_time =
      1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(linear_solver_end_time - linear_solver_start_time).count();
    linearization_time = std::chrono::nanoseconds(0);

    if (params_.status_msg_callback) {
      const std::string msg = status.to_string();
      const auto newline_loc = msg.find('\n');

      if (newline_loc == std::string::npos) {
        params_.status_msg_callback(msg);
      } else {
        params_.status_msg_callback(msg.substr(0, newline_loc));
        params_.status_msg_callback(msg.substr(newline_loc + 1));
      }
    }

    if (params_.callback) {
      params_.callback(status, currentState->values);
    }
  }

  if (step_is_successful) {
    // we have successfully decreased the cost and we have good modelFidelity
    // NOTE(frank): As we return immediately after this, we move the newValues
    // TODO(frank): make Values actually support move. Does not seem to happen now.
    state_ = currentState->decreaseLambda(params_, modelFidelity, std::move(newValues), newError);
    return true;
  } else if (!stopSearchingLambda) {         // we failed to solved the system or had no decrease in cost
    State* modifiedState = static_cast<State*>(state_.get());
    modifiedState->increaseLambda(params_);  // TODO(frank): make this functional with Values move

    // check if lambda is too big
    if (modifiedState->lambda >= params_.lambdaUpperBound) {
      std::cout << "Warning:  Levenberg-Marquardt giving up because cannot decrease error with maximum lambda" << std::endl;
      return true;
    } else {
      return false;  // only case where we will keep trying
    }
  } else {           // the change in the cost is very small and it is not worth trying bigger lambdas
    return true;
  }
}

gtsam::GaussianFactorGraph::shared_ptr LevenbergMarquardtOptimizerExt::iterate() {
  auto currentState = static_cast<const State*>(state_.get());

  auto t1 = std::chrono::high_resolution_clock::now();
  gtsam::GaussianFactorGraph::shared_ptr linear;
  linearization_hook->linearize(currentState->values);
  linear = graph_.linearize(currentState->values);
  auto t2 = std::chrono::high_resolution_clock::now();
  linearization_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);

  linearization_hook->error(currentState->values);
  double oldError = graph_.error(currentState->values);
  // double oldError = graph_.error(currentState->values, linear);

  gtsam::VectorValues sqrtHessianDiagonal;
  if (params_.diagonalDamping) {
    sqrtHessianDiagonal = linear->hessianDiagonal();
    for (gtsam::Vector& v : sqrtHessianDiagonal | boost::adaptors::map_values) {
      v = v.cwiseMax(params_.minDiagonal).cwiseMin(params_.maxDiagonal).cwiseSqrt();
    }
  }

  while (!tryLambda(*linear, sqrtHessianDiagonal, oldError)) {
    // auto newState = static_cast<const State*>(state_.get());
  }

  // graph_ext_->post_optimization(state_->values);
  return linear;
}

const gtsam::Values& LevenbergMarquardtOptimizerExt::optimize() {
  optimization_start_time = std::chrono::high_resolution_clock::now();
  // defaultOptimize();

  const gtsam::NonlinearOptimizerParams& params = _params();

  bool terminate = false;
  double currentError = state_->error;

  do {
    linearized = iterate();
    double newError = state_->error;

    terminate |= iterations() > params.maxIterations;
    terminate |= !std::isfinite(newError);

    // error can increase due to data association changes
    // terminate |= gtsam::checkConvergence(params.relativeErrorTol, params.absoluteErrorTol, params.errorTol, currentError, newError,
    // params.verbosity);

    double delta_error = std::abs(currentError - newError);
    terminate |= delta_error < params.absoluteErrorTol;
    terminate |= delta_error / std::abs(currentError) < params.relativeErrorTol;
    terminate |= newError < params.errorTol;

    if (params_.termination_criteria) {
      const auto state = static_cast<const State*>(state_.get());
      terminate |= params_.termination_criteria(values());
    }

    currentError = newError;
  } while (!terminate);

  return values();
}
}  // namespace gtsam_points
