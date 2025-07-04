/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    DoglegOptimizer.cpp
 * @brief
 * @author  Richard Roberts
 * @date   Feb 26, 2012
 */

#include <gtsam_points/optimizers/dogleg_optimizer_ext.hpp>
#include <gtsam_points/optimizers/dogleg_optimizer_ext_impl.hpp>

#include <gtsam/nonlinear/internal/NonlinearOptimizerState.h>
#include <gtsam/linear/GaussianBayesTree.h>
#include <gtsam/linear/GaussianBayesNet.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/VectorValues.h>

namespace gtsam_points {

using namespace gtsam;

/* ************************************************************************* */
namespace internal {
struct DoglegState : public gtsam::internal::NonlinearOptimizerState {
  const double delta;

  DoglegState(const Values& _values, double _error, double _delta, unsigned int _iterations = 0)
  : NonlinearOptimizerState(_values, _error, _iterations),
    delta(_delta) {}
};
}  // namespace internal

typedef internal::DoglegState State;

/* ************************************************************************* */
DoglegOptimizerExt::DoglegOptimizerExt(const NonlinearFactorGraph& graph, const Values& initialValues, const DoglegParams& params)
: NonlinearOptimizer(graph, std::unique_ptr<State>(new State(initialValues, graph.error(initialValues), params.deltaInitial))),
  params_(ensureHasOrdering(params, graph)) {}

DoglegOptimizerExt::DoglegOptimizerExt(const NonlinearFactorGraph& graph, const Values& initialValues, const Ordering& ordering)
: NonlinearOptimizer(graph, std::unique_ptr<State>(new State(initialValues, graph.error(initialValues), 1.0))) {
  params_.ordering = ordering;
}

double DoglegOptimizerExt::getDelta() const {
  return static_cast<const State*>(state_.get())->delta;
}

/* ************************************************************************* */
GaussianFactorGraph::shared_ptr DoglegOptimizerExt::iterate(void) {
  // Linearize graph
  GaussianFactorGraph::shared_ptr linear = graph_.linearize(state_->values);

  // Pull out parameters we'll use
  const bool dlVerbose = (params_.verbosityDL > DoglegParams::SILENT);

  // Do Dogleg iteration with either Multifrontal or Sequential elimination
  DoglegOptimizerImplExt::IterationResult result;

  if (params_.isMultifrontal()) {
    GaussianBayesTree bt = *linear->eliminateMultifrontal(*params_.ordering, params_.getEliminationFunction());
    VectorValues dx_u = bt.optimizeGradientSearch();
    VectorValues dx_n = bt.optimize();
    result = DoglegOptimizerImplExt::Iterate(
      getDelta(),
      DoglegOptimizerImplExt::ONE_STEP_PER_ITERATION,
      dx_u,
      dx_n,
      bt,
      graph_,
      state_->values,
      state_->error,
      dlVerbose);
  } else if (params_.isSequential()) {
    GaussianBayesNet bn = *linear->eliminateSequential(*params_.ordering, params_.getEliminationFunction());
    VectorValues dx_u = bn.optimizeGradientSearch();
    VectorValues dx_n = bn.optimize();
    result = DoglegOptimizerImplExt::Iterate(
      getDelta(),
      DoglegOptimizerImplExt::ONE_STEP_PER_ITERATION,
      dx_u,
      dx_n,
      bn,
      graph_,
      state_->values,
      state_->error,
      dlVerbose);
  } else if (params_.isIterative()) {
    throw std::runtime_error("Dogleg is not currently compatible with the linear conjugate gradient solver");
  } else {
    throw std::runtime_error("Optimization parameter is invalid: DoglegParams::elimination");
  }

  // Maybe show output
  if (params_.verbosity >= NonlinearOptimizerParams::DELTA) result.dx_d.print("delta");

  // Create new state with new values and new error
  state_.reset(new State(state_->values.retract(result.dx_d), result.f_error, result.delta, state_->iterations + 1));
  return linear;
}

/* ************************************************************************* */
DoglegParams DoglegOptimizerExt::ensureHasOrdering(DoglegParams params, const NonlinearFactorGraph& graph) const {
  if (!params.ordering) params.ordering = Ordering::Create(params.orderingType, graph);
  return params;
}

}  // namespace gtsam_points
