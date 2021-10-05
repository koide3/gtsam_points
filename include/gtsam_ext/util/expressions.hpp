#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/expressions.h>
#include <gtsam/nonlinear/expressions.h>

namespace gtsam_ext {

inline gtsam::Pose3_ expmap(const gtsam::Vector6_& x) {
  return gtsam::Pose3_(&gtsam::Pose3::Expmap, x);
}

inline gtsam::Vector6_ logmap(const gtsam::Pose3_& x) {
  return gtsam::Vector6_(&gtsam::Pose3::Logmap, x);
}

inline gtsam::Pose3_ inverse(const gtsam::Pose3_& x) {
  auto f = [](const gtsam::Pose3& x, gtsam::OptionalJacobian<6, 6> H) {
    return x.inverse(H);
  };
  return gtsam::Pose3_(f, x);
}

}