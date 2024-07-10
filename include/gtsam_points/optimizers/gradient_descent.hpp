// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <chrono>
#include <functional>
#include <boost/format.hpp>

#include <gtsam/linear/VectorValues.h>
#include <gtsam/nonlinear/NonlinearOptimizer.h>
#include <gtsam/nonlinear/internal/NonlinearOptimizerState.h>

namespace gtsam_points {

struct NumericalGradientDescentStatus {
  std::string to_string() const { return (boost::format("%d: error:%.6f -> %.6f lambda:%10g") % iterations % error_old % error_new % lambda).str(); }

  int iterations;
  double lambda;
  double error_old;
  double error_new;
};

class NumericalGradientDescentParams : public gtsam::NonlinearOptimizerParams {
public:
  NumericalGradientDescentParams() : gtsam::NonlinearOptimizerParams(), max_iterations(1024), diff_eps(1e-6), init_lambda(1e-10), lambda_rate(2.0) {}

  int max_iterations;
  double diff_eps;

  double init_lambda;
  double lambda_rate;

  std::function<void(const NumericalGradientDescentStatus&, const gtsam::Values&)> callback;  // callback for optimization iteration
};

struct NumericalGradientDescentState : public gtsam::internal::NonlinearOptimizerState {
  NumericalGradientDescentState(
    const NumericalGradientDescentParams& params,
    double lambda,
    const gtsam::Values& initial_values,
    double errors = std::numeric_limits<double>::max(),
    int iterations = 0)
  : gtsam::internal::NonlinearOptimizerState(initial_values, errors, iterations),
    params(params),
    lambda(lambda) {}

  NumericalGradientDescentState* decrease_lambda(const gtsam::Values& values, double error) const {
    return new NumericalGradientDescentState(params, lambda / params.lambda_rate, values, error, iterations + 1);
  }

  NumericalGradientDescentState* increase_lambda(const gtsam::Values& values, double error) const {
    return new NumericalGradientDescentState(params, lambda * params.lambda_rate, values, error, iterations + 1);
  }

  const NumericalGradientDescentParams& params;
  double lambda;
};

/**
 * @brief Naive gradient descent with numerical differentiation for objective function test
 *        Never use this for practical application!!
 */
class NumericalGradientDescent : public gtsam::NonlinearOptimizer {
public:
  using State = NumericalGradientDescentState;

  NumericalGradientDescent(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& initial_value, const NumericalGradientDescentParams& params)
  : gtsam::NonlinearOptimizer(graph, std::unique_ptr<NumericalGradientDescentState>(new NumericalGradientDescentState(params, params.init_lambda, initial_value))),
    params_(params) {}

  virtual gtsam::GaussianFactorGraph::shared_ptr iterate() override {
    const auto& current_state = static_cast<const NumericalGradientDescentState*>(state_.get());

    const auto& x0 = state_->values;
    const auto linearized = graph_.linearize(x0);
    const double y0 = graph_.error(x0);

    const int dim = x0.dim();
    gtsam::VectorValues J = x0.zeroVectors();

    for (const auto key : x0.keys()) {
      for (int i = 0; i < J[key].size(); i++) {
        gtsam::VectorValues delta = gtsam::VectorValues::Zero(J);
        delta[key][i] = params_.diff_eps;

        const gtsam::Values xi = x0.retract(delta);
        const double yi = graph_.error(xi);

        J[key][i] = (yi - y0) / params_.diff_eps;
      }
    }

    const double lambda = current_state->lambda;
    const gtsam::VectorValues delta = J.scale(-lambda);
    const gtsam::Values xi = x0.retract(delta);
    const double yi = graph_.error(xi);
    const double cost_change = yi - y0;

    if (yi < y0) {
      auto new_state = current_state->increase_lambda(xi, yi);
      state_.reset(new_state);
    } else {
      auto new_state = current_state->decrease_lambda(x0, y0);
      state_.reset(new_state);
    }

    if (params_.callback) {
      NumericalGradientDescentStatus status;
      status.iterations = state_->iterations;
      status.lambda = lambda;
      status.error_old = y0;
      status.error_new = yi;

      params_.callback(status, state_->values);
    }

    return nullptr;
  }

  virtual const gtsam::Values& optimize() override {
    for (int i = 0; i < params_.max_iterations; i++) {
      iterate();
    }

    return state_->values;
  }

  const gtsam::NonlinearOptimizerParams& _params() const override { return params_; }

private:
  const NumericalGradientDescentParams params_;

  std::chrono::high_resolution_clock::time_point optimization_start_time;
};

}  // namespace gtsam_points