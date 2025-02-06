// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/ExpressionFactor.h>

namespace gtsam_points {

class RotateVector3Factor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Vector3> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Base = gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Vector3>;

  RotateVector3Factor() {}
  RotateVector3Factor(gtsam::Key x, gtsam::Key v, const gtsam::Vector3& local_v, const gtsam::SharedNoiseModel& noise_model)
  : Base(noise_model, x, v),
    local_v(local_v) {}

  virtual ~RotateVector3Factor() {}

  virtual void print(const std::string& s = "", const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const override {
    std::cout << s << "RotateVector3Factor";
    std::cout << "(" << keyFormatter(this->keys()[0]) << ", " << keyFormatter(this->keys()[1]) << ")" << std::endl;
  }

  virtual gtsam::Vector evaluateError(
    const gtsam::Pose3& x,
    const gtsam::Vector3& v,
    boost::optional<gtsam::Matrix&> H_x,
    boost::optional<gtsam::Matrix&> H_v) const override {
    gtsam::Matrix36 H_x1;
    gtsam::Matrix33 H_x2;
    gtsam::Vector3 v_ = x.rotation(H_x1).rotate(local_v, H_x2);
    gtsam::Vector3 error = v_ - v;

    if (H_x) {
      (*H_x) = H_x2 * H_x1;
    }

    if (H_v) {
      (*H_v) = -gtsam::Matrix33::Identity();
    }

    return error;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override { return gtsam::NonlinearFactor::shared_ptr(new RotateVector3Factor(*this)); }

private:
  /** Serialization function */
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& boost::serialization::make_nvp("NoiseModelFactor2", boost::serialization::base_object<Base>(*this));
    ar& BOOST_SERIALIZATION_NVP(local_v);
  }

private:
  gtsam::Vector3 local_v;
};

}  // namespace gtsam_points