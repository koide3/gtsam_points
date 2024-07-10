// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <gtsam/linear/GaussianFactorGraph.h>

namespace gtsam_points {

class DenseLinearSolver;
class SparseLinearSolver;

class GaussianFactorGraphSolver {
public:
  GaussianFactorGraphSolver() {}
  virtual ~GaussianFactorGraphSolver() {}

  virtual gtsam::VectorValues solve(const gtsam::GaussianFactorGraph& gfg, const gtsam::Ordering& ordering) = 0;
};

class DenseGaussianFactorGraphSolver : public GaussianFactorGraphSolver {
public:
  DenseGaussianFactorGraphSolver(const std::shared_ptr<DenseLinearSolver>& solver);
  virtual ~DenseGaussianFactorGraphSolver() override;

  virtual gtsam::VectorValues solve(const gtsam::GaussianFactorGraph& gfg, const gtsam::Ordering& ordering) override;

private:
  std::shared_ptr<DenseLinearSolver> solver;
};

class SparseGaussianFactorGraphSolver : public GaussianFactorGraphSolver {
public:
  SparseGaussianFactorGraphSolver(const std::shared_ptr<SparseLinearSolver>& solver);
  virtual ~SparseGaussianFactorGraphSolver() override;

  virtual gtsam::VectorValues solve(const gtsam::GaussianFactorGraph& gfg, const gtsam::Ordering& ordering) override;

private:
  std::shared_ptr<SparseLinearSolver> solver;
};

class GTSAMGaussianFactorGraphSolver : public GaussianFactorGraphSolver {
public:
  GTSAMGaussianFactorGraphSolver();
  virtual ~GTSAMGaussianFactorGraphSolver() override;

  virtual gtsam::VectorValues solve(const gtsam::GaussianFactorGraph& gfg, const gtsam::Ordering& ordering) override;
};

}  // namespace gtsam_points
