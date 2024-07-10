// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <string>

namespace gtsam_points {

struct LevenbergMarquardtOptimizationStatus {
public:
  std::string to_string() const;
  std::string to_short_string() const;

public:
  int iterations;                // number of iterations
  int total_inner_iterations;    // number of LM lambda trials

  double error;                  // error_t
  double cost_change;            // error_t - error_{t-1}
  double lambda;                 // current lambda for LM
  bool solve_success;            // linear solver succeeded

  double elapsed_time;           // time since optimization beginning
  double linearization_time;     // time spent for linearization
  double lambda_iteration_time;  // time since LM iteration beginning
  double linear_solver_time;     // time spent for solving the linear system
};

}  // namespace gtsam_points
