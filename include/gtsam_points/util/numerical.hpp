// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <Eigen/Core>

namespace gtsam_points {

template <typename Func>
Eigen::VectorXd numerical_jacobian(const Func& f, const Eigen::VectorXd& x, double eps = 1e-6) {
  const int N = x.size();
  Eigen::VectorXd j(N);

  for (int i = 0; i < N; i++) {
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(N);
    dx[i] = eps;

    const double y0 = f(x - dx);
    const double y1 = f(x + dx);
    j[i] = (y1 - y0) / (2.0 * eps);
  }

  return j;
}

template <typename Func>
Eigen::MatrixXd numerical_hessian(const Func& f, const Eigen::VectorXd& x, double eps = 1e-6) {
  const int N = x.size();
  Eigen::MatrixXd h(N, N);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      Eigen::VectorXd dx = Eigen::VectorXd::Zero(N);
      dx[i] = eps;

      auto first = [&](const Eigen::VectorXd& dy) {
        const double y0 = f(x - dx + dy);
        const double y1 = f(x + dx + dy);
        return (y1 - y0) / (2.0 * eps);
      };

      Eigen::VectorXd dy = Eigen::VectorXd::Zero(N);
      dy[j] = eps;

      const double dx0 = first(-dy);
      const double dx1 = first(dy);

      h(i, j) = (dx1 - dx0) / (2.0 * eps);
    }
  }

  return h;
}
}  // namespace gtsam_points