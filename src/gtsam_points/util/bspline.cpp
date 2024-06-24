// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/util/bspline.hpp>

#include <gtsam_points/util/expressions.hpp>

namespace gtsam_points {

gtsam::Pose3_
bspline(const gtsam::Pose3_& pose0, const gtsam::Pose3_& pose1, const gtsam::Pose3_& pose2, const gtsam::Pose3_& pose3, const gtsam::Double_& t) {
  const gtsam::Double_ t2 = t * t;
  const gtsam::Double_ t3 = t2 * t;

  const gtsam::Double_ beta1 = gtsam::Double_(5.0 / 6.0) + 3.0 / 6.0 * t - 3.0 / 6.0 * t2 + 1.0 / 6.0 * t3;
  const gtsam::Double_ beta2 = gtsam::Double_(1.0 / 6.0) + 3.0 / 6.0 * t + 3.0 / 6.0 * t2 - 2.0 / 6.0 * t3;
  const gtsam::Double_ beta3 = 1.0 / 6.0 * t3;

  const gtsam::Rot3_ rot0 = gtsam::rotation(pose0);
  const gtsam::Rot3_ rot1 = gtsam::rotation(pose1);
  const gtsam::Rot3_ rot2 = gtsam::rotation(pose2);
  const gtsam::Rot3_ rot3 = gtsam::rotation(pose3);

  const gtsam::Rot3_ r_delta1 = gtsam_points::expmap(gtsam_points::scale(beta1, gtsam::logmap(rot0, rot1)));
  const gtsam::Rot3_ r_delta2 = gtsam_points::expmap(gtsam_points::scale(beta2, gtsam::logmap(rot1, rot2)));
  const gtsam::Rot3_ r_delta3 = gtsam_points::expmap(gtsam_points::scale(beta3, gtsam::logmap(rot2, rot3)));

  const gtsam::Vector3_ trans0 = gtsam::translation(pose0);
  const gtsam::Vector3_ trans1 = gtsam::translation(pose1);
  const gtsam::Vector3_ trans2 = gtsam::translation(pose2);
  const gtsam::Vector3_ trans3 = gtsam::translation(pose3);

  const gtsam::Vector3_ t_delta1 = gtsam_points::scale(beta1, gtsam::between(trans0, trans1));
  const gtsam::Vector3_ t_delta2 = gtsam_points::scale(beta2, gtsam::between(trans1, trans2));
  const gtsam::Vector3_ t_delta3 = gtsam_points::scale(beta3, gtsam::between(trans2, trans3));

  const gtsam::Rot3_ rot = gtsam::compose(gtsam::compose(gtsam::compose(rot0, r_delta1), r_delta2), r_delta3);
  const gtsam::Vector3_ trans = gtsam::compose(gtsam::compose(gtsam::compose(trans0, t_delta1), t_delta2), t_delta3);
  const gtsam::Pose3_ pose = gtsam_points::create_se3(rot, trans);

  return pose;
}

gtsam::Pose3_
bspline_se3(const gtsam::Pose3_& pose0, const gtsam::Pose3_& pose1, const gtsam::Pose3_& pose2, const gtsam::Pose3_& pose3, const gtsam::Double_& t) {
  const gtsam::Double_ t2 = t * t;
  const gtsam::Double_ t3 = t2 * t;

  const gtsam::Double_ beta1 = gtsam::Double_(5.0 / 6.0) + 3.0 / 6.0 * t - 3.0 / 6.0 * t2 + 1.0 / 6.0 * t3;
  const gtsam::Double_ beta2 = gtsam::Double_(1.0 / 6.0) + 3.0 / 6.0 * t + 3.0 / 6.0 * t2 - 2.0 / 6.0 * t3;
  const gtsam::Double_ beta3 = 1.0 / 6.0 * t3;

  const gtsam::Pose3_ delta1 = gtsam_points::expmap(gtsam_points::scale(beta1, gtsam::logmap(pose0, pose1)));
  const gtsam::Pose3_ delta2 = gtsam_points::expmap(gtsam_points::scale(beta2, gtsam::logmap(pose1, pose2)));
  const gtsam::Pose3_ delta3 = gtsam_points::expmap(gtsam_points::scale(beta3, gtsam::logmap(pose2, pose3)));

  const gtsam::Pose3_ pose = gtsam::compose(gtsam::compose(gtsam::compose(pose0, delta1), delta2), delta3);

  return pose;
}

gtsam::Rot3_
bspline_so3(const gtsam::Rot3_& rot0, const gtsam::Rot3_& rot1, const gtsam::Rot3_& rot2, const gtsam::Rot3_& rot3, const gtsam::Double_& t) {
  const gtsam::Double_ t2 = t * t;
  const gtsam::Double_ t3 = t2 * t;

  const gtsam::Double_ beta1 = gtsam::Double_(5.0 / 6.0) + 3.0 / 6.0 * t - 3.0 / 6.0 * t2 + 1.0 / 6.0 * t3;
  const gtsam::Double_ beta2 = gtsam::Double_(1.0 / 6.0) + 3.0 / 6.0 * t + 3.0 / 6.0 * t2 - 2.0 / 6.0 * t3;
  const gtsam::Double_ beta3 = 1.0 / 6.0 * t3;

  const gtsam::Rot3_ r_delta1 = gtsam_points::expmap(gtsam_points::scale(beta1, gtsam::logmap(rot0, rot1)));
  const gtsam::Rot3_ r_delta2 = gtsam_points::expmap(gtsam_points::scale(beta2, gtsam::logmap(rot1, rot2)));
  const gtsam::Rot3_ r_delta3 = gtsam_points::expmap(gtsam_points::scale(beta3, gtsam::logmap(rot2, rot3)));

  const gtsam::Rot3_ rot = gtsam::compose(gtsam::compose(gtsam::compose(rot0, r_delta1), r_delta2), r_delta3);

  return rot;
}

gtsam::Vector3_ bspline_trans(
  const gtsam::Vector3_& trans0,
  const gtsam::Vector3_& trans1,
  const gtsam::Vector3_& trans2,
  const gtsam::Vector3_& trans3,
  const gtsam::Double_& t) {
  const gtsam::Double_ t2 = t * t;
  const gtsam::Double_ t3 = t2 * t;

  const gtsam::Double_ beta1 = gtsam::Double_(5.0 / 6.0) + 3.0 / 6.0 * t - 3.0 / 6.0 * t2 + 1.0 / 6.0 * t3;
  const gtsam::Double_ beta2 = gtsam::Double_(1.0 / 6.0) + 3.0 / 6.0 * t + 3.0 / 6.0 * t2 - 2.0 / 6.0 * t3;
  const gtsam::Double_ beta3 = 1.0 / 6.0 * t3;

  const gtsam::Vector3_ t_delta1 = gtsam_points::scale(beta1, gtsam::between(trans0, trans1));
  const gtsam::Vector3_ t_delta2 = gtsam_points::scale(beta2, gtsam::between(trans1, trans2));
  const gtsam::Vector3_ t_delta3 = gtsam_points::scale(beta3, gtsam::between(trans2, trans3));

  const gtsam::Vector3_ trans = gtsam::compose(gtsam::compose(gtsam::compose(trans0, t_delta1), t_delta2), t_delta3);

  return trans;
}

gtsam::Vector3_ bspline_angular_vel(
  const gtsam::Rot3_& rot0,
  const gtsam::Rot3_& rot1,
  const gtsam::Rot3_& rot2,
  const gtsam::Rot3_& rot3,
  const gtsam::Double_& t,
  const double knot_interval) {
  const gtsam::Double_ t2 = t * t;
  const gtsam::Double_ t3 = t2 * t;

  const gtsam::Double_ beta1 = gtsam::Double_(5.0 / 6.0) + 3.0 / 6.0 * t - 3.0 / 6.0 * t2 + 1.0 / 6.0 * t3;
  const gtsam::Double_ beta2 = gtsam::Double_(1.0 / 6.0) + 3.0 / 6.0 * t + 3.0 / 6.0 * t2 - 2.0 / 6.0 * t3;
  const gtsam::Double_ beta3 = 1.0 / 6.0 * t3;

  const gtsam::Double_ H_beta1_t = gtsam::Double_(3.0 / 6.0) - 3.0 / 6.0 * 2.0 * t + 1.0 / 6.0 * 3.0 * t2;
  const gtsam::Double_ H_beta2_t = gtsam::Double_(3.0 / 6.0) + 3.0 / 6.0 * 2.0 * t - 2.0 / 6.0 * 3.0 * t2;
  const gtsam::Double_ H_beta3_t = 1.0 / 6.0 * 3.0 * t2;

  const gtsam::Vector3_ d1 = gtsam::logmap(rot0, rot1);
  const gtsam::Vector3_ d2 = gtsam::logmap(rot1, rot2);
  const gtsam::Vector3_ d3 = gtsam::logmap(rot2, rot3);

  const gtsam::Rot3_ A1 = gtsam_points::expmap(gtsam_points::scale(beta1, d1));
  const gtsam::Rot3_ A2 = gtsam_points::expmap(gtsam_points::scale(beta2, d2));
  const gtsam::Rot3_ A3 = gtsam_points::expmap(gtsam_points::scale(beta3, d3));

  const gtsam::Vector3_ omega1 = gtsam_points::scale(H_beta1_t, d1);
  const gtsam::Vector3_ omega2 = gtsam_points::add(gtsam::unrotate(A1, omega1), gtsam_points::scale(H_beta2_t, d2));
  const gtsam::Vector3_ omega3 = gtsam_points::add(gtsam::unrotate(A2, omega2), gtsam_points::scale(H_beta3_t, d3));

  return (1.0 / knot_interval) * omega3;
}

gtsam::Vector3_ bspline_linear_vel(
  const gtsam::Vector3_& trans0,
  const gtsam::Vector3_& trans1,
  const gtsam::Vector3_& trans2,
  const gtsam::Vector3_& trans3,
  const gtsam::Double_& t,
  const double knot_interval) {
  //
  const gtsam::Double_ t2 = t * t;
  const gtsam::Double_ t3 = t2 * t;

  const gtsam::Double_ beta1 = gtsam::Double_(5.0 / 6.0) + 3.0 / 6.0 * t - 3.0 / 6.0 * t2 + 1.0 / 6.0 * t3;
  const gtsam::Double_ beta2 = gtsam::Double_(1.0 / 6.0) + 3.0 / 6.0 * t + 3.0 / 6.0 * t2 - 2.0 / 6.0 * t3;
  const gtsam::Double_ beta3 = 1.0 / 6.0 * t3;

  const gtsam::Double_ H_beta1_t = gtsam::Double_(3.0 / 6.0) - 3.0 / 6.0 * 2.0 * t + 1.0 / 6.0 * 3.0 * t2;
  const gtsam::Double_ H_beta2_t = gtsam::Double_(3.0 / 6.0) + 3.0 / 6.0 * 2.0 * t - 2.0 / 6.0 * 3.0 * t2;
  const gtsam::Double_ H_beta3_t = 1.0 / 6.0 * 3.0 * t2;

  const gtsam::Vector3_ d1 = gtsam::between(trans0, trans1);
  const gtsam::Vector3_ d2 = gtsam::between(trans1, trans2);
  const gtsam::Vector3_ d3 = gtsam::between(trans2, trans3);

  const gtsam::Vector3_ omega1 = gtsam_points::scale(H_beta1_t, d1);
  const gtsam::Vector3_ omega2 = gtsam::compose(omega1, gtsam_points::scale(H_beta2_t, d2));
  const gtsam::Vector3_ omega3 = gtsam::compose(omega2, gtsam_points::scale(H_beta3_t, d3));

  return (1.0 / knot_interval) * omega3;
}

gtsam::Vector3_ bspline_linear_acc(
  const gtsam::Vector3_& trans0,
  const gtsam::Vector3_& trans1,
  const gtsam::Vector3_& trans2,
  const gtsam::Vector3_& trans3,
  const gtsam::Double_& t,
  const double knot_interval) {
  //
  const gtsam::Double_ t2 = t * t;
  const gtsam::Double_ t3 = t2 * t;

  const gtsam::Double_ beta1 = gtsam::Double_(5.0 / 6.0) + 3.0 / 6.0 * t - 3.0 / 6.0 * t2 + 1.0 / 6.0 * t3;
  const gtsam::Double_ beta2 = gtsam::Double_(1.0 / 6.0) + 3.0 / 6.0 * t + 3.0 / 6.0 * t2 - 2.0 / 6.0 * t3;
  const gtsam::Double_ beta3 = 1.0 / 6.0 * t3;

  const gtsam::Double_ H_beta1_t = gtsam::Double_(3.0 / 6.0) - 3.0 / 6.0 * 2.0 * t + 1.0 / 6.0 * 3.0 * t2;
  const gtsam::Double_ H_beta2_t = gtsam::Double_(3.0 / 6.0) + 3.0 / 6.0 * 2.0 * t - 2.0 / 6.0 * 3.0 * t2;
  const gtsam::Double_ H_beta3_t = 1.0 / 6.0 * 3.0 * t2;

  const gtsam::Double_ H2_beta1_t = gtsam::Double_(-3.0 / 6.0 * 2.0) + 1.0 / 6.0 * 3.0 * 2.0 * t;
  const gtsam::Double_ H2_beta2_t = gtsam::Double_(3.0 / 6.0 * 2.0) - 2.0 / 6.0 * 3.0 * 2.0 * t;
  const gtsam::Double_ H2_beta3_t = 1.0 / 6.0 * 3.0 * 2.0 * t;

  const gtsam::Vector3_ d1 = gtsam::between(trans0, trans1);
  const gtsam::Vector3_ d2 = gtsam::between(trans1, trans2);
  const gtsam::Vector3_ d3 = gtsam::between(trans2, trans3);

  const gtsam::Vector3_ omega1_ = gtsam_points::scale(H2_beta1_t, d1);
  const gtsam::Vector3_ omega2_ = gtsam::compose(omega1_, gtsam_points::scale(H2_beta2_t, d2));
  const gtsam::Vector3_ omega3_ = gtsam::compose(omega2_, gtsam_points::scale(H2_beta3_t, d3));

  return (1.0 / (knot_interval * knot_interval)) * omega3_;
}

gtsam::Vector6_ bspline_imu(
  const gtsam::Pose3_ pose0,
  const gtsam::Pose3_ pose1,
  const gtsam::Pose3_ pose2,
  const gtsam::Pose3_ pose3,
  const gtsam::Double_& t,
  const double knot_interval,
  const gtsam::Vector3& g) {
  //
  const gtsam::Double_ t2 = t * t;
  const gtsam::Double_ t3 = t2 * t;

  // Weights and their temporal derivatives
  const gtsam::Double_ beta1 = gtsam::Double_(5.0 / 6.0) + 3.0 / 6.0 * t - 3.0 / 6.0 * t2 + 1.0 / 6.0 * t3;
  const gtsam::Double_ beta2 = gtsam::Double_(1.0 / 6.0) + 3.0 / 6.0 * t + 3.0 / 6.0 * t2 - 2.0 / 6.0 * t3;
  const gtsam::Double_ beta3 = 1.0 / 6.0 * t3;

  const gtsam::Double_ H_beta1_t = gtsam::Double_(3.0 / 6.0) - 3.0 / 6.0 * 2.0 * t + 1.0 / 6.0 * 3.0 * t2;
  const gtsam::Double_ H_beta2_t = gtsam::Double_(3.0 / 6.0) + 3.0 / 6.0 * 2.0 * t - 2.0 / 6.0 * 3.0 * t2;
  const gtsam::Double_ H_beta3_t = 1.0 / 6.0 * 3.0 * t2;

  const gtsam::Double_ H2_beta1_t = gtsam::Double_(-3.0 / 6.0 * 2.0) + 1.0 / 6.0 * 3.0 * 2.0 * t;
  const gtsam::Double_ H2_beta2_t = gtsam::Double_(3.0 / 6.0 * 2.0) - 2.0 / 6.0 * 3.0 * 2.0 * t;
  const gtsam::Double_ H2_beta3_t = 1.0 / 6.0 * 3.0 * 2.0 * t;

  // Rotation
  const gtsam::Rot3_ rot0 = gtsam::rotation(pose0);
  const gtsam::Rot3_ rot1 = gtsam::rotation(pose1);
  const gtsam::Rot3_ rot2 = gtsam::rotation(pose2);
  const gtsam::Rot3_ rot3 = gtsam::rotation(pose3);

  const gtsam::Vector3_ r_d1 = gtsam::logmap(rot0, rot1);
  const gtsam::Vector3_ r_d2 = gtsam::logmap(rot1, rot2);
  const gtsam::Vector3_ r_d3 = gtsam::logmap(rot2, rot3);

  const gtsam::Rot3_ r_A1 = gtsam_points::expmap(gtsam_points::scale(beta1, r_d1));
  const gtsam::Rot3_ r_A2 = gtsam_points::expmap(gtsam_points::scale(beta2, r_d2));
  const gtsam::Rot3_ r_A3 = gtsam_points::expmap(gtsam_points::scale(beta3, r_d3));

  const gtsam::Vector3_ r_omega1 = gtsam_points::scale(H_beta1_t, r_d1);
  const gtsam::Vector3_ r_omega2 = gtsam_points::add(gtsam::unrotate(r_A1, r_omega1), gtsam_points::scale(H_beta2_t, r_d2));
  const gtsam::Vector3_ r_omega3 = gtsam_points::add(gtsam::unrotate(r_A2, r_omega2), gtsam_points::scale(H_beta3_t, r_d3));

  // Translation
  const gtsam::Vector3_ trans0 = gtsam::translation(pose0);
  const gtsam::Vector3_ trans1 = gtsam::translation(pose1);
  const gtsam::Vector3_ trans2 = gtsam::translation(pose2);
  const gtsam::Vector3_ trans3 = gtsam::translation(pose3);

  const gtsam::Vector3_ t_d1 = gtsam::between(trans0, trans1);
  const gtsam::Vector3_ t_d2 = gtsam::between(trans1, trans2);
  const gtsam::Vector3_ t_d3 = gtsam::between(trans2, trans3);

  const gtsam::Vector3_ t_omega1_ = gtsam_points::scale(H2_beta1_t, t_d1);
  const gtsam::Vector3_ t_omega2_ = gtsam::compose(t_omega1_, gtsam_points::scale(H2_beta2_t, t_d2));
  const gtsam::Vector3_ t_omega3_ = gtsam::compose(t_omega2_, gtsam_points::scale(H2_beta3_t, t_d3));

  // Transform in the local coordinate
  const gtsam::Rot3_ rot = gtsam::compose(gtsam::compose(gtsam::compose(rot0, r_A1), r_A2), r_A3);

  const double inv_knot_interval = 1.0 / knot_interval;
  const double inv_knot_interval2 = inv_knot_interval * inv_knot_interval;

  const gtsam::Vector3_ angular_vel = inv_knot_interval * r_omega3;
  const gtsam::Vector3_ linear_acc = gtsam::unrotate(rot, inv_knot_interval2 * t_omega3_ + gtsam::Vector3_(g));

  return gtsam_points::concatenate(linear_acc, angular_vel);
}
}  // namespace gtsam_points