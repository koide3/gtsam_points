// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace gtsam_ext {

class DenseLinearSolver {
public:
  virtual ~DenseLinearSolver() {}

  virtual Eigen::VectorXd solve(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) = 0;
};

class SparseLinearSolver {
public:
  virtual ~SparseLinearSolver() {}

  virtual Eigen::VectorXd solve(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A, const Eigen::VectorXd& b) = 0;
};

}  // namespace gtsam_ext
