// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/optimizers/levenberg_marquardt_optimization_status.hpp>

#include <sstream>
#include <boost/format.hpp>

namespace gtsam_points {

std::string LevenbergMarquardtOptimizationStatus::to_string() const {
  std::stringstream sst;

  char check[] = {' ', 'x'};

  bool dec = cost_change > 0.0;

  double lin_sec = linearization_time;

  // clang-format off
  if(iterations == 0 && total_inner_iterations == 0) {
    sst << boost::format("%5s %5s %15s %15s %10s %5s %5s %10s %10s %10s %10s\n") % "iter" % "lm" % "error" % "cost_d" % "lambda" % "ldec" % "dec" % "lin_msec" % "solv_msec" % "lm_msec" % "time_msec";
  }

  sst << boost::format("%5d %5d %15g %15g %10g %5c %5c %10g %10g %10g %10g") % iterations % total_inner_iterations % error % cost_change % lambda % check[solve_success] % check[dec]  % (lin_sec * 1000) % (linear_solver_time * 1000) % (lambda_iteration_time * 1000) % (elapsed_time * 1000);
  // clang-format on

  return sst.str();
}

std::string LevenbergMarquardtOptimizationStatus::to_short_string() const {
  std::stringstream sst;

  char check[] = {' ', 'x'};

  bool dec = cost_change > 0.0;

  // clang-format off
  if(iterations == 0 && total_inner_iterations == 0) {
    sst << boost::format("%3s %3s %15s %10s %5s %10s\n") % "iter" % "lm" % "error" % "lambda" % "dec" % "time_msec";
  }

  sst << boost::format("%3d %3d %15g %10g %5c %10g") % iterations % total_inner_iterations % error % lambda % check[dec]  % (elapsed_time * 1000);
  // clang-format on

  return sst.str();
}

}  // namespace gtsam_points
