#pragma once

#include <string>

namespace gtsam_ext {

struct LevenbergMarquardtOptimizationStatus {
public:
  std::string to_string() const;
  std::string to_short_string() const;

public:
  int iterations;              // number of iterations
  int total_inner_iterations;  // number of LM lambda trials

  double error;        // error_t
  double cost_change;  // error_t - error_{t-1}
  double lambda;       // current lambda for LM
  bool solve_success;  // linear solver succeeded

  double elapsed_time;           // time since optimization beginning
  double lambda_iteration_time;  // time since LM iteration beginning
};

}  // namespace gtsam_ext
