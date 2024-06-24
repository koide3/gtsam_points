// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Similarity3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam_points {

gtsam::Pose3 scaled_transform(const gtsam::Similarity3& sim3, gtsam::OptionalJacobian<6, 7> H = boost::none) {
  if (H) {
    H->setZero();
    H->block<3, 3>(0, 0).setIdentity();
    H->block<3, 3>(3, 3) = gtsam::Matrix3::Identity() * sim3.scale();
    // H->block<3, 1>(3, 6) = sim3.translation();
  }

  return gtsam::Pose3(sim3.rotation(), sim3.scale() * sim3.translation());
}

class BetweenSim3SE3Factor : public gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Pose3> {
public:
  BetweenSim3SE3Factor(const gtsam::Key x1_key, const gtsam::Key x2_key, const gtsam::SharedNoiseModel& noise_model)
  : gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Pose3>(noise_model, x1_key, x2_key) {}

  virtual gtsam::Vector evaluateError(
    const gtsam::Similarity3& x1,
    const gtsam::Pose3& x2,
    boost::optional<gtsam::Matrix&> H1 = boost::none,
    boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
    //
    if (!H1 || !H2) {
      gtsam::Pose3 delta = scaled_transform(x1).between(x2);
      return gtsam::Pose3::Logmap(delta);
    }

    gtsam::Matrix67 H_scaled_x1;
    gtsam::Pose3 scaled = scaled_transform(x1, H_scaled_x1);

    gtsam::Matrix66 H_delta_scaled, H_delta_x2;
    gtsam::Pose3 delta = scaled.between(x2, H_delta_scaled, H_delta_x2);

    gtsam::Matrix6 H_error_delta;
    gtsam::Vector6 error = gtsam::Pose3::Logmap(delta, H_error_delta);

    if (H1 && H2) {
      (*H1) = H_error_delta * H_delta_scaled * H_scaled_x1;
      (*H2) = H_error_delta * H_delta_x2;
    }

    return error;
  }
};
}  // namespace gtsam_points