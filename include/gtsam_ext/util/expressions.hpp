#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/expressions.h>
#include <gtsam/nonlinear/expressions.h>

namespace gtsam_ext {

gtsam::Pose3_ expmap(const gtsam::Vector6_& x) {
  return gtsam::Pose3_(&gtsam::Pose3::Expmap, x);
}

gtsam::Vector6_ logmap(const gtsam::Pose3_& x) {
  return gtsam::Vector6_(&gtsam::Pose3::Logmap, x);
}

gtsam::Pose3_ inverse(const gtsam::Pose3_& x) {
  auto f = [](const gtsam::Pose3& x, gtsam::OptionalJacobian<6, 6> H) {
    return x.inverse(H);
  };
  return gtsam::Pose3_(f, x);
}

gtsam::Point3_ cross(const gtsam::Point3_& lhs, const gtsam::Point3_& rhs) {
  return gtsam::Point3_(gtsam::cross, lhs, rhs);
}

}