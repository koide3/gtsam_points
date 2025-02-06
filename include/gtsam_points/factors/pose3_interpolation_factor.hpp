// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam_points {

/**
 * @brief Factor(xi, xj, xk) s.t. xk = Slerp(xi, xj, t)
 */
class Pose3InterpolationFactor : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {
public:
  Pose3InterpolationFactor(gtsam::Key xi, gtsam::Key xj, gtsam::Key xk, const double t, const gtsam::SharedNoiseModel& noise_model)
  : gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(noise_model, xi, xj, xk),
    t(t) {}

  virtual ~Pose3InterpolationFactor() override {}

  virtual void print(const std::string& s = "", const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const override {
    std::cout << s << "Pose3InterpolationFactor";
    std::cout << "(" << keyFormatter(this->keys()[0]) << ", " << keyFormatter(this->keys()[1]) << ", " << keyFormatter(this->keys()[2]) << ")"
              << std::endl;
  }

public:
  virtual gtsam::Vector evaluateError(
    const gtsam::Pose3& xi,
    const gtsam::Pose3& xj,
    const gtsam::Pose3& xk,
    boost::optional<gtsam::Matrix&> H_xi = boost::none,
    boost::optional<gtsam::Matrix&> H_xj = boost::none,
    boost::optional<gtsam::Matrix&> H_xk = boost::none) const override {
    //
    const auto& Ri = xi.rotation();
    const auto& Rj = xj.rotation();
    const auto& Rk = xk.rotation();

    const auto& ti = xi.translation();
    const auto& tj = xj.translation();
    const auto& tk = xk.translation();

    if (!H_xi) {
      const gtsam::Rot3 Rint = gtsam::interpolate(Ri, Rj, t);
      const gtsam::Rot3 Re = Rk.between(Rint);
      const gtsam::Vector3 re = gtsam::Rot3::Logmap(Re);
      const gtsam::Vector3 te = tk - (1.0 - t) * ti - t * tj;

      return (gtsam::Vector6() << re, te).finished();
    }

    gtsam::Matrix33 H_Rint_Ri, H_Rint_Rj;
    const gtsam::Rot3 Rint = gtsam::interpolate(Ri, Rj, t, H_Rint_Ri, H_Rint_Rj);

    // H_Re_Rint == Identity
    gtsam::Matrix33 H_Re_Rk;
    const gtsam::Rot3 Re = Rk.between(Rint, H_Re_Rk);

    gtsam::Matrix33 H_re_Re;
    const gtsam::Vector3 re = gtsam::Rot3::Logmap(Re, H_re_Re);

    const gtsam::Vector3 te = tk - (1.0 - t) * ti - t * tj;

    (*H_xi) = gtsam::Matrix6::Zero();
    (*H_xj) = gtsam::Matrix6::Zero();
    (*H_xk) = gtsam::Matrix6::Zero();

    // const gtsam::Matrix3 H_re_Rint = H_re_Re * H_Re_Rint;
    const auto& H_re_Rint = H_re_Re;
    H_xi->block<3, 3>(0, 0) = H_re_Rint * H_Rint_Ri;
    H_xj->block<3, 3>(0, 0) = H_re_Rint * H_Rint_Rj;
    H_xk->block<3, 3>(0, 0) = H_re_Re * H_Re_Rk;

    H_xi->block<3, 3>(3, 3) = -(1.0 - t) * Ri.matrix();
    H_xj->block<3, 3>(3, 3) = -t * Rj.matrix();
    H_xk->block<3, 3>(3, 3) = Rk.matrix();

    return (gtsam::Vector6() << re, te).finished();
  }

  static gtsam::Pose3 initial_guess(const gtsam::Pose3& xi, const gtsam::Pose3& xj, const double t) {
    const gtsam::Rot3 Rint = gtsam::interpolate(xi.rotation(), xj.rotation(), t);
    const gtsam::Vector3 tint = (1.0 - t) * xi.translation() + t * xj.translation();
    return gtsam::Pose3(Rint, tint);
  }

private:
  const double t;
};

}  // namespace gtsam_points