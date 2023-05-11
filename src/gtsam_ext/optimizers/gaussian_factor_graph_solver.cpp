#include <gtsam_ext/optimizers/gaussian_factor_graph_solver.hpp>

#include <gtsam_ext/optimizers/linear_solver.hpp>
#include <gtsam_ext/optimizers/linear_system_builder.hpp>

namespace gtsam_ext {

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
  return system.delta(solver->solve(system.A, system.b));
}

GTSAMGaussianFactorGraphSolver::GTSAMGaussianFactorGraphSolver() {}

GTSAMGaussianFactorGraphSolver::~GTSAMGaussianFactorGraphSolver() {}

gtsam::VectorValues GTSAMGaussianFactorGraphSolver::solve(const gtsam::GaussianFactorGraph& gfg, const gtsam::Ordering& ordering) {
  // return gfg.eliminateMultifrontal()->optimize(ordering);
  return gfg.optimize(ordering);
}

}  // namespace gtsam_ext
