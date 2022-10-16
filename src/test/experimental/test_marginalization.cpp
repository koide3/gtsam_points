#include <iostream>
#include <unordered_set>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/inference/Ordering.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

class ShurMarginalizer {
public:
  void marginalize(gtsam::NonlinearFactorGraph& factors, gtsam::Values& values, const gtsam::KeyVector& keys_to_eliminate) const {
    if (keys_to_eliminate.empty()) {
      return;
    }

    std::unordered_set<gtsam::Key> keyset_to_eliminate(keys_to_eliminate.begin(), keys_to_eliminate.end());

    auto factors_to_eliminate = find_factors_to_eliminate(factors, keyset_to_eliminate);
    auto linearized_factors = factors_to_eliminate.linearize(values);

    gtsam::Ordering ordering = gtsam::Ordering::ColamdConstrainedLast(factors_to_eliminate, keys_to_eliminate);

    std::vector<int> dims;
    std::vector<int> start_locs;
    for (int i = 0; i < ordering.size() - keys_to_eliminate.size(); i++) {
      start_locs.push_back(dims.empty() ? 0 : start_locs.back() + dims.back());
      dims.push_back(values.at(ordering[i]).dim());
    }

    const int remaining_dim = start_locs.back() + dims.back();
    const int eliminated_dim = values.dim() - remaining_dim;

    auto Hg = linearized_factors->hessian(ordering);
    const auto& H = Hg.first;
    const auto& g = Hg.second;

    const auto& H11 = H.topLeftCorner(remaining_dim, remaining_dim);
    const auto& H12 = H.topRightCorner(remaining_dim, eliminated_dim);
    const auto& H21 = H.bottomLeftCorner(eliminated_dim, remaining_dim);
    const auto& H22 = H.bottomRightCorner(eliminated_dim, eliminated_dim);
    const auto& g1 = g.topRows(remaining_dim);
    const auto& g2 = g.bottomRows(eliminated_dim);

    // Schur complement
    // should use sparse operations?
    gtsam::Matrix M = H11 - H12 * H22.selfadjointView<Eigen::Upper>().ldlt().solve(H21);
    gtsam::Vector b = g1 - H12 * H22.selfadjointView<Eigen::Upper>().ldlt().solve(g2);

    for (const auto& key : keys_to_eliminate) {
      values.erase(key);
    }
  }

  gtsam::NonlinearFactorGraph find_factors_to_eliminate(
    gtsam::NonlinearFactorGraph& factors,
    const std::unordered_set<gtsam::Key>& keyset_to_eliminate) const {
    gtsam::NonlinearFactorGraph factors_to_eliminate;
    for (int i = 0; i < factors.size(); i++) {
      const auto& factor = factors.at(i);
      if (factor == nullptr) {
        continue;
      }

      if (!std::any_of(factor->begin(), factor->end(), [&](const gtsam::Key& key) { return keyset_to_eliminate.count(key); })) {
        continue;
      }

      factors_to_eliminate.push_back(factor);
      factors.erase(factors.begin() + i);
      i--;
    }

    return factors_to_eliminate;
  }
};

int main(int argc, char** argv) {
  gtsam::Matrix36 H = gtsam::Matrix36::Random();

  Eigen::FullPivLU<gtsam::Matrix36> lu(H);
  gtsam::Matrix null_space = lu.kernel();

  std::cout << "--- H ---" << std::endl << H << std::endl;

  std::cout << "--- null_space ---" << std::endl << null_space << std::endl;

  std::cout << "--- null ---" << std::endl << H * null_space << std::endl;

  return 0;

  auto factor = gtsam::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(0, 1, gtsam::Pose3(), gtsam::noiseModel::Isotropic::Precision(6, 1e3));

  gtsam::Values values;
  values.insert(0, gtsam::Pose3::Expmap(gtsam::Vector6::Random()));
  values.insert(1, gtsam::Pose3::Expmap(gtsam::Vector6::Random()));

  auto linearized = factor->linearize(values);

  auto inf = linearized->augmentedInformation();

  std::cout << inf << std::endl;
  std::cout << inf.rows() << " " << inf.cols() << std::endl;

  return 0;
}