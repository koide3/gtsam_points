// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <random>
#include <vector>
#include <gtsam/linear/VectorValues.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam_points {

/**
 * @brief Utility function for testing Jacobian of a factor.
 * @note  This function cannot properly evaluate Jacobian of factors that measures errors in a manifold (e.g., BetweenFactor<Pose3>).
 *
 * @example

auto generate_factor(std::mt19937& mt) {
  std::uniform_real_distribution<> udist(-1.0, 1.0);

  gtsam::Vector3 rel_pose = Eigen::Vector3d(udist(mt), udist(mt), udist(mt));
  return gtsam::make_shared<gtsam::BetweenFactor<gtsam::Vector3>>(0, 1, rel_pose, gtsam::noiseModel::Isotropic::Sigma(6, 1.0));
}

gtsam::Values generate_values(std::mt19937& mt) {
  std::uniform_real_distribution<> udist(-1.0, 1.0);

  gtsam::Values values;
  values.insert(0, Eigen::Vector3d(udist(mt), udist(mt), udist(mt)));
  values.insert(1, Eigen::Vector3d(udist(mt), udist(mt), udist(mt)));
  return values;
}

int main(int argc, char** argv) {
  std::mt19937 mt;
  for (int i = 0; i < 5; i++) {
    gtsam_points::test_factor_jacobian(mt, generate_factor, generate_values);
  }
  return 0;
}

 */
template <typename GenerateFactor, typename GenerateValues>
bool test_factor_jacobian(std::mt19937& mt, const GenerateFactor& generate_factor, const GenerateValues& generate_values) {
  auto factor = generate_factor(mt);
  gtsam::Values values = generate_values(mt);

  std::vector<gtsam::Matrix> Hs_a(factor->keys().size());
  std::vector<gtsam::Matrix> Hs_d(factor->keys().size());
  gtsam::Vector x0 = factor->unwhitenedError(values, Hs_a);

  const double eps = 1e-6;

  for (int i = 0; i < factor->keys().size(); i++) {
    const auto key = factor->keys()[i];
    gtsam::VectorValues delta = values.zeroVectors();
    const int D = delta[key].size();

    gtsam::Matrix Hd(x0.size(), D);
    for (int j = 0; j < D; j++) {
      delta[key].setZero();
      delta[key][j] = eps;

      gtsam::Vector xi = factor->unwhitenedError(values.retract(delta));
      Hd.col(j) = (xi - x0) / eps;
    }

    Hs_d[i] = Hd;
  }

  bool ok = true;
  for (int i = 0; i < factor->keys().size(); i++) {
    ok &= ((Hs_a[i] - Hs_d[i]).array().abs() < 1e-3).all();
  }

  if (ok) {
    std::cout << "OK (x0=" << x0.transpose() << ")" << std::endl;
  } else {
    std::cout << "Failed (x0=" << x0.transpose() << ")" << std::endl;

    for (int i = 0; i < factor->keys().size(); i++) {
      std::cout << "*** key " << i << " ***" << std::endl;
      std::cerr << "--- Ha ---" << std::endl << Hs_a[i] << std::endl;
      std::cerr << "--- Hd ---" << std::endl << Hs_d[i] << std::endl;
      std::cerr << "--- err ---" << std::endl << Hs_a[i] - Hs_d[i] << std::endl;
    }
  }

  return ok;
}

}  // namespace gtsam_points
