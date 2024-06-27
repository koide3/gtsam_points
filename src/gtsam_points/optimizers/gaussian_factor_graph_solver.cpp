// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/optimizers/gaussian_factor_graph_solver.hpp>

#include <gtsam_points/optimizers/linear_solver.hpp>
#include <gtsam_points/optimizers/linear_system_builder.hpp>

#include <gtsam_points/util/easy_profiler.hpp>

namespace gtsam_points {

DenseGaussianFactorGraphSolver::DenseGaussianFactorGraphSolver(const std::shared_ptr<DenseLinearSolver>& solver) : solver(solver) {}

DenseGaussianFactorGraphSolver::~DenseGaussianFactorGraphSolver() {}

gtsam::VectorValues DenseGaussianFactorGraphSolver::solve(const gtsam::GaussianFactorGraph& gfg, const gtsam::Ordering& ordering) {
  DenseLinearSystemBuilder system(gfg, ordering);
  return system.delta(solver->solve(system.A, system.b));
}

SparseGaussianFactorGraphSolver::SparseGaussianFactorGraphSolver(const std::shared_ptr<SparseLinearSolver>& solver) : solver(solver) {}

SparseGaussianFactorGraphSolver::~SparseGaussianFactorGraphSolver() {}

gtsam::VectorValues SparseGaussianFactorGraphSolver::solve(const gtsam::GaussianFactorGraph& gfg, const gtsam::Ordering& ordering) {
  SparseLinearSystemBuilder system(gfg, ordering);
  auto x = solver->solve(system.A, system.b);
  return system.delta(x);
}

GTSAMGaussianFactorGraphSolver::GTSAMGaussianFactorGraphSolver() {}

GTSAMGaussianFactorGraphSolver::~GTSAMGaussianFactorGraphSolver() {}

gtsam::VectorValues GTSAMGaussianFactorGraphSolver::solve(const gtsam::GaussianFactorGraph& gfg, const gtsam::Ordering& ordering) {
  // return gfg.eliminateMultifrontal()->optimize(ordering);
  return gfg.optimize(ordering);
}

}  // namespace gtsam_points
