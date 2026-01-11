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

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/internal/LevenbergMarquardtState.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam/inference/Ordering.h>
#include <gtsam/base/Vector.h>
#if GTSAM_USE_BOOST_FEATURES
#include <gtsam/base/timing.h>
#endif

#include <gtsam_points/config.hpp>
#include <gtsam_points/util/parallelism.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/parallel_for.h>
#endif

#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>
#include <numeric>

using namespace std;

namespace gtsam_points {

typedef gtsam::internal::LevenbergMarquardtState State;

using gtsam::GaussianFactorGraph;
using gtsam::IndeterminantLinearSystemException;
using gtsam::LevenbergMarquardtParams;
using gtsam::NonlinearFactorGraph;
using gtsam::NonlinearOptimizerParams;
using gtsam::Ordering;
using gtsam::Values;
using gtsam::VectorValues;

double calc_error(const gtsam::GaussianFactorGraph& gfg, const gtsam::VectorValues& x) {
  if (is_omp_default()) {
    return gfg.error(x);
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    // TODO: Should use parallel reduction
    std::vector<double> errors(gfg.size(), 0.0);
    tbb::parallel_for(static_cast<size_t>(0), gfg.size(), [&](size_t i) { errors[i] = gfg[i]->error(x); });
    return std::accumulate(errors.begin(), errors.end(), 0.0);
#else
    std::cerr << "error: gtsam_points is not built with TBB!!" << std::endl;
    return gfg.error(x);
#endif
  }
}

double calc_error(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& x) {
  if (is_omp_default()) {
    return graph.error(x);
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    // TODO: Should use parallel reduction
    std::vector<double> errors(graph.size(), 0.0);
    tbb::parallel_for(static_cast<size_t>(0), graph.size(), [&](size_t i) { errors[i] = graph[i]->error(x); });
    return std::accumulate(errors.begin(), errors.end(), 0.0);
#else
    std::cerr << "error: gtsam_points is not built with TBB!!" << std::endl;
    return graph.error(x);
#endif
  }
}

/* ************************************************************************* */
LevenbergMarquardtOptimizerExt::LevenbergMarquardtOptimizerExt(
  const NonlinearFactorGraph& graph,
  const Values& initialValues,
  const LevenbergMarquardtExtParams& params)
: NonlinearOptimizer(
    graph,
    std::unique_ptr<State>(new State(initialValues, std::numeric_limits<double>::max(), params.lambdaInitial, params.lambdaFactor))),
  params_(LevenbergMarquardtExtParams::EnsureHasOrdering(params, graph)),
  linearization_hook_(new LinearizationHook(graph)) {}

LevenbergMarquardtOptimizerExt::LevenbergMarquardtOptimizerExt(
  const NonlinearFactorGraph& graph,
  const Values& initialValues,
  const Ordering& ordering,
  const LevenbergMarquardtExtParams& params)
: NonlinearOptimizer(
    graph,
    std::unique_ptr<State>(new State(initialValues, std::numeric_limits<double>::max(), params.lambdaInitial, params.lambdaFactor))),
  params_(LevenbergMarquardtExtParams::ReplaceOrdering(params, ordering)),
  linearization_hook_(new LinearizationHook(graph)) {}

LevenbergMarquardtOptimizerExt::~LevenbergMarquardtOptimizerExt() {}

/* ************************************************************************* */
void LevenbergMarquardtOptimizerExt::initTime() {
  // use chrono to measure time in microseconds
  startTime_ = std::chrono::high_resolution_clock::now();
}

/* ************************************************************************* */
double LevenbergMarquardtOptimizerExt::lambda() const {
  auto currentState = static_cast<const State*>(state_.get());
  return currentState->lambda;
}

/* ************************************************************************* */
int LevenbergMarquardtOptimizerExt::getInnerIterations() const {
  auto currentState = static_cast<const State*>(state_.get());
  return currentState->totalNumberInnerIterations;
}

/* ************************************************************************* */
GaussianFactorGraph::shared_ptr LevenbergMarquardtOptimizerExt::linearize() const {
  gtsam::GaussianFactorGraph::shared_ptr linear;
  linearization_hook_->linearize(state_->values);
  linear = graph_.linearize(state_->values);
  return linear;
}

/* ************************************************************************* */
GaussianFactorGraph LevenbergMarquardtOptimizerExt::buildDampedSystem(const GaussianFactorGraph& linear, const VectorValues& sqrtHessianDiagonal)
  const {
  gttic(damp);
  auto currentState = static_cast<const State*>(state_.get());

  if (params_.verbosityLM >= LevenbergMarquardtParams::DAMPED)
    std::cout << "building damped system with lambda " << currentState->lambda << std::endl;

  if (params_.diagonalDamping)
    return currentState->buildDampedSystem(linear, sqrtHessianDiagonal);
  else
    return currentState->buildDampedSystem(linear);
}

/* ************************************************************************* */
// Log current error/lambda to file
inline void LevenbergMarquardtOptimizerExt::writeLogFile(double currentError) {
  auto currentState = static_cast<const State*>(state_.get());

  if (!params_.logFile.empty()) {
    ofstream os(params_.logFile.c_str(), ios::app);
    // use chrono to measure time in microseconds
    auto currentTime = std::chrono::high_resolution_clock::now();
    // Get the time spent in seconds and print it
    double timeSpent = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - startTime_).count() / 1e6;
    os << /*inner iterations*/ currentState->totalNumberInnerIterations << "," << timeSpent << "," << /*current error*/ currentError << ","
       << currentState->lambda << "," << /*outer iterations*/ currentState->iterations << endl;
  }
}

/* ************************************************************************* */
bool LevenbergMarquardtOptimizerExt::tryLambda(const GaussianFactorGraph& linear, const VectorValues& sqrtHessianDiagonal) {
  auto lambda_iteration_start_time = std::chrono::high_resolution_clock::now();

  auto currentState = static_cast<const State*>(state_.get());
  bool verbose = (params_.verbosityLM >= LevenbergMarquardtParams::TRYLAMBDA);

#if GTSAM_USE_BOOST_FEATURES
#ifdef GTSAM_USING_NEW_BOOST_TIMERS
  boost::timer::cpu_timer lamda_iteration_timer;
  lamda_iteration_timer.start();
#else
  boost::timer lamda_iteration_timer;
  lamda_iteration_timer.restart();
#endif
#else
  auto start = std::chrono::high_resolution_clock::now();
#endif

  if (verbose) cout << "trying lambda = " << currentState->lambda << endl;

  // Build damped system for this lambda (adds prior factors that make it like gradient descent)
  auto dampedSystem = buildDampedSystem(linear, sqrtHessianDiagonal);

  // Try solving
  double modelFidelity = 0.0;
  bool step_is_successful = false;
  bool stopSearchingLambda = false;
  double newError = numeric_limits<double>::infinity();
  double costChange = 0.0;
  Values newValues;
  VectorValues delta;

  auto linear_solver_start_time = std::chrono::high_resolution_clock::now();
  bool systemSolvedSuccessfully;
  try {
    // ============ Solve is where most computation happens !! =================
    delta = solve(dampedSystem, params_);
    systemSolvedSuccessfully = true;
  } catch (const IndeterminantLinearSystemException&) {
    systemSolvedSuccessfully = false;
  }
  auto linear_solver_end_time = std::chrono::high_resolution_clock::now();

  if (systemSolvedSuccessfully) {
    if (verbose) cout << "linear delta norm = " << delta.norm() << endl;
    if (params_.verbosityLM >= LevenbergMarquardtParams::TRYDELTA) delta.print("delta");

    // Compute the old linearized error as it is not the same
    // as the nonlinear error when robust noise models are used.
    double oldLinearizedError = calc_error(linear, gtsam::VectorValues::Zero(delta));
    double newlinearizedError = calc_error(linear, delta);
    // cost change in the linearized system (old - new)
    double linearizedCostChange = oldLinearizedError - newlinearizedError;
    if (verbose) cout << "newlinearizedError = " << newlinearizedError << "  linearizedCostChange = " << linearizedCostChange << endl;

    if (linearizedCostChange >= 0) {  // step is valid
      // update values
      gttic(retract);
      // ============ This is where the solution is updated ====================
      newValues = currentState->values.retract(delta);
      // =======================================================================
      gttoc(retract);

      // compute new error
      gttic(compute_error);
      if (verbose) cout << "calculating error:" << endl;
      linearization_hook_->error(newValues);
      newError = calc_error(graph_, newValues);
      gttoc(compute_error);

      if (verbose) cout << "old error (" << currentState->error << ") new (tentative) error (" << newError << ")" << endl;

      // cost change in the original, nonlinear system (old - new)
      costChange = currentState->error - newError;

      if (linearizedCostChange > std::numeric_limits<double>::epsilon() * oldLinearizedError) {
        // the (linear) error has to decrease to satisfy this condition
        // fidelity of linearized model VS original system between
        modelFidelity = costChange / linearizedCostChange;
        // if we decrease the error in the nonlinear system and modelFidelity is above threshold
        step_is_successful = modelFidelity > params_.minModelFidelity;
        if (verbose) cout << "modelFidelity: " << modelFidelity << endl;
      }  // else we consider the step non successful and we either increase lambda or stop if error
         // change is small

      double minAbsoluteTolerance = params_.relativeErrorTol * currentState->error;
      // if the change is small we terminate
      if (std::abs(costChange) < minAbsoluteTolerance) {
        if (verbose)
          cout << "abs(costChange)=" << std::abs(costChange) << "  minAbsoluteTolerance=" << minAbsoluteTolerance
               << " (relativeErrorTol=" << params_.relativeErrorTol << ")" << endl;
        stopSearchingLambda = true;
      }
    }
  }  // if (systemSolvedSuccessfully)
  if (params_.callback || params_.status_msg_callback) {
    LevenbergMarquardtOptimizationStatus status;
    status.iterations = currentState->iterations;
    status.total_inner_iterations = currentState->totalNumberInnerIterations;
    status.error = currentState->error;
    status.cost_change = costChange;
    status.lambda = currentState->lambda;
    status.solve_success = systemSolvedSuccessfully;

    auto t = std::chrono::high_resolution_clock::now();
    status.elapsed_time = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(t - optimization_start_time_).count();
    status.linearization_time = 1e-9 * linearization_time_.count();
    status.lambda_iteration_time = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(t - lambda_iteration_start_time).count();
    status.linear_solver_time =
      1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(linear_solver_end_time - linear_solver_start_time).count();
    linearization_time_ = std::chrono::nanoseconds(0);

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

  if (params_.verbosityLM == LevenbergMarquardtParams::SUMMARY) {
#if GTSAM_USE_BOOST_FEATURES
// do timing
#ifdef GTSAM_USING_NEW_BOOST_TIMERS
    double iterationTime = 1e-9 * lamda_iteration_timer.elapsed().wall;
#else
    double iterationTime = lamda_iteration_timer.elapsed();
#endif
#else
    auto end = std::chrono::high_resolution_clock::now();
    double iterationTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6;
#endif
    if (currentState->iterations == 0) {
      cout << "iter      cost      cost_change    lambda  success iter_time" << endl;
    }
    cout << setw(4) << currentState->iterations << " " << setw(12) << newError << " " << setw(12) << setprecision(2) << costChange << " " << setw(10)
         << setprecision(2) << currentState->lambda << " " << setw(6) << systemSolvedSuccessfully << " " << setw(10) << setprecision(2)
         << iterationTime << endl;
  }
  if (step_is_successful) {
    // we have successfully decreased the cost and we have good modelFidelity
    // NOTE(frank): As we return immediately after this, we move the newValues
    // TODO(frank): make Values actually support move. Does not seem to happen now.
    state_ = currentState->decreaseLambda(params_, modelFidelity, std::move(newValues), newError);
    return true;
  } else if (!stopSearchingLambda) {  // we failed to solved the system or had no decrease in cost
    State* modifiedState = static_cast<State*>(state_.get());
    modifiedState->increaseLambda(params_);  // TODO(frank): make this functional with Values move

    // check if lambda is too big
    if (modifiedState->lambda >= params_.lambdaUpperBound) {
      if (params_.verbosity >= NonlinearOptimizerParams::TERMINATION || params_.verbosityLM == LevenbergMarquardtParams::SUMMARY)
        cout << "Warning:  Levenberg-Marquardt giving up because "
                "cannot decrease error with maximum lambda"
             << endl;
      return true;
    } else {
      return false;  // only case where we will keep trying
    }
  } else {  // the change in the cost is very small and it is not worth trying bigger lambdas
    return true;
  }
}

/* ************************************************************************* */
GaussianFactorGraph::shared_ptr LevenbergMarquardtOptimizerExt::iterate() {
  auto currentState = static_cast<State*>(state_.get());

  gttic(LM_iterate);

  // Linearize graph
  if (params_.verbosityLM >= LevenbergMarquardtParams::DAMPED) cout << "linearizing = " << endl;
  auto t1 = std::chrono::high_resolution_clock::now();
  linearization_hook_->linearize(state_->values);
  gtsam::GaussianFactorGraph::shared_ptr linear = graph_.linearize(state_->values);
  auto t2 = std::chrono::high_resolution_clock::now();
  linearization_time_ = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);

  // I'm not sure the following is correct, come back to it later
  // const double err = 2.0 * linear->error(state_->values.zeroVectors());
  // const_cast<double&>(currentState->error) = err;  // !!!

  linearization_hook_->error(currentState->values);
  const_cast<double&>(currentState->error) = graph_.error(currentState->values);  // !! Bad practice

  if (currentState->totalNumberInnerIterations == 0) {  // write initial erroro
    writeLogFile(currentState->error);

    if (params_.verbosityLM == LevenbergMarquardtParams::SUMMARY) {
      cout << "Initial error: " << currentState->error << ", values: " << currentState->values.size() << std::endl;
    }
  }

  // Only calculate diagonal of Hessian (expensive) once per outer iteration, if we need it
  VectorValues sqrtHessianDiagonal;
  if (params_.diagonalDamping) {
    sqrtHessianDiagonal = linear->hessianDiagonal();
    for (auto& [key, value] : sqrtHessianDiagonal) {
      value = value.cwiseMax(params_.minDiagonal).cwiseMin(params_.maxDiagonal).cwiseSqrt();
    }
  }

  // Keep increasing lambda until we make make progress
  while (!tryLambda(*linear, sqrtHessianDiagonal)) {
    auto newState = static_cast<const State*>(state_.get());
    writeLogFile(newState->error);
  }

  return linear;
}

const gtsam::Values& LevenbergMarquardtOptimizerExt::optimize() {
  optimization_start_time_ = std::chrono::high_resolution_clock::now();
  // defaultOptimize();

  const gtsam::NonlinearOptimizerParams& params = _params();

  bool terminate = false;
  double currentError = state_->error;

  do {
    linearized_ = iterate();
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
