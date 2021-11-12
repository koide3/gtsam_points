// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/ExpressionFactor.h>

namespace gtsam {

class RotateVector3Factor : public NoiseModelFactor2<Pose3, Vector3> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Base = NoiseModelFactor2<Pose3, Vector3>;

  RotateVector3Factor() {}
  RotateVector3Factor(Key x, Key v, const Vector3& local_v, const SharedNoiseModel& noise_model) : Base(noise_model, x, v), local_v(local_v) {}

  virtual ~RotateVector3Factor() {}

  virtual Vector evaluateError(const Pose3& x, const Vector3& v, boost::optional<Matrix&> H_x, boost::optional<Matrix&> H_v) const override {
    Matrix36 H_x1;
    Matrix33 H_x2;
    Vector3 v_ = x.rotation(H_x1).rotate(local_v, H_x2);
    Vector3 error = v_ - v;

    if (H_x) {
      (*H_x) = H_x2 * H_x1;
    }

    if (H_v) {
      (*H_v) = -Matrix33::Identity();
    }

    return error;
  }

private:
  Vector3 local_v;
};

}  // namespace gtsam