// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam_points {

/**
 * @brief Factor(xi, xj, Tij) that evaluates (xj_inv * xi) * Tij
 */
class Pose3CalibFactor : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {
public:
  Pose3CalibFactor(gtsam::Key xi, gtsam::Key xj, gtsam::Key Tij, const gtsam::SharedNoiseModel& noise_model)
  : gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(noise_model, xi, xj, Tij) {}

  virtual ~Pose3CalibFactor() override {}

  virtual void print(const std::string& s = "", const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const override {
    std::cout << s << "Pose3CalibFactor";
    std::cout << "(" << keyFormatter(this->keys()[0]) << ", " << keyFormatter(this->keys()[1]) << ", " << keyFormatter(this->keys()[2]) << ")"
              << std::endl;
  }

public:
  virtual gtsam::Vector evaluateError(
    const gtsam::Pose3& xi,
    const gtsam::Pose3& xj,
    const gtsam::Pose3& Tij,
    boost::optional<gtsam::Matrix&> H_xi = boost::none,
    boost::optional<gtsam::Matrix&> H_xj = boost::none,
    boost::optional<gtsam::Matrix&> H_Tij = boost::none) const override {
    //
    if (H_xi == boost::none && H_xj == boost::none && H_Tij == boost::none) {
      const gtsam::Pose3 delta = xi.between(xj);      // (xi^-1 * xj)
      const gtsam::Pose3 error = delta.between(Tij);  // (xj_inv * xi) * Tij;
      return gtsam::Pose3::Logmap(error);
    }

    gtsam::Matrix6 H_delta_i;
    gtsam::Matrix6 H_delta_j;
    gtsam::Pose3 delta = xi.between(xj, H_delta_i, H_delta_j);

    gtsam::Matrix6 H_error_delta;
    gtsam::Matrix6 H_error_Tij;
    gtsam::Pose3 error = delta.between(Tij, H_error_delta, H_error_Tij);

    gtsam::Matrix6 H_log_error;
    gtsam::Vector6 log = gtsam::Pose3::Logmap(error, H_log_error);

    (*H_xi) = H_log_error * H_error_delta * H_delta_i;
    (*H_xj) = H_log_error * H_error_delta * H_delta_j;
    (*H_Tij) = H_log_error * H_error_Tij;

    return log;
  }
};

}  // namespace gtsam_points