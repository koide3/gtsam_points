// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam/slam/expressions.h>

namespace gtsam_points {

/**
 * @brief B-Spline pose interpolation
 *        Rotation and translation are independently interpolated
 * @note  Requirement: t0 < t1 < t2 < t3 and t is the normalized time between t1 and t2 in [0, 1]
 *        Sec. 2.2 in https://www.robots.ox.ac.uk/~mobile/Theses/StewartThesis.pdf
 * @param pose0 Pose at t0
 * @param pose1 Pose at t1
 * @param pose2 Pose at t2
 * @param pose3 Pose at t3
 * @param t     Normalized time between t1 and t2 in [0, 1]
 * @return      Interpolated pose
 */
gtsam::Pose3_ bspline(const gtsam::Pose3_& pose0, const gtsam::Pose3_& pose1, const gtsam::Pose3_& pose2, const gtsam::Pose3_& pose3, const gtsam::Double_& t);

/**
 * @brief B-Spline pose interpolation
 *        Rotation and translation are jointly interpolated
 *        This would be suitable for interpolating twist motion (e.g., vehicle motion)
 * @note  Requirement: t0 < t1 < t2 < t3 and t is the normalized time between t1 and t2 in [0, 1]
 *        Sec. 2.2 in https://www.robots.ox.ac.uk/~mobile/Theses/StewartThesis.pdf
 * @param pose0 Pose at t0
 * @param pose1 Pose at t1
 * @param pose2 Pose at t2
 * @param pose3 Pose at t3
 * @param t     Normalized time between t1 and t2 in [0, 1]
 * @return      Interpolated pose
 */
gtsam::Pose3_ bspline_se3(const gtsam::Pose3_& pose0, const gtsam::Pose3_& pose1, const gtsam::Pose3_& pose2, const gtsam::Pose3_& pose3, const gtsam::Double_& t);

/**
 * @brief B-Spline rotation interpolation
 * @note  Requirement: t0 < t1 < t2 < t3 and t is the normalized time between t1 and t2 in [0, 1]
 *        Sec. 2.2 in https://www.robots.ox.ac.uk/~mobile/Theses/StewartThesis.pdf
 * @param rot0  Rotation at t0
 * @param rot1  Rotation at t1
 * @param rot2  Rotation at t2
 * @param rot3  Rotation at t3
 * @param t     Normalized time between t1 and t2 in [0, 1]
 * @return      Interpolated pose
 */
gtsam::Rot3_ bspline_so3(const gtsam::Rot3_& rot0, const gtsam::Rot3_& rot1, const gtsam::Rot3_& rot2, const gtsam::Rot3_& rot3, const gtsam::Double_& t);

/**
 * @brief B-Spline translation interpolation
 * @note  Requirement: t0 < t1 < t2 < t3 and t is the normalized time between t1 and t2 in [0, 1]
 *        Sec. 2.2 in https://www.robots.ox.ac.uk/~mobile/Theses/StewartThesis.pdf
 * @param trans0  Translation at t0
 * @param trans1  Translation at t1
 * @param trans2  Translation at t2
 * @param trans3  Translation at t3
 * @param t       Normalized time between t1 and t2 in [0, 1]
 * @return        Interpolated pose
 */
gtsam::Vector3_ bspline_trans(const gtsam::Vector3_& trans0, const gtsam::Vector3_& trans1, const gtsam::Vector3_& trans2, const gtsam::Vector3_& trans3, const gtsam::Double_& t);

/**
 * @brief Calculate global angular velocity of B-spline interpolated trajectory
 *        Sommer et al., "Efficient Derivative Computation for Cumulative B-Splines on Lie Groups", CVPR2020
 * @param rot0          Rotation at t0
 * @param rot1          Rotation at t1
 * @param rot2          Rotation at t2
 * @param rot3          Rotation at t3
 * @param t             Normalized time between t1 and t2 in [0, 1]
 * @param knot_interval Real time interval between spline knots
 * @return              Global angular velocity of the interpolated trajectory
 */
gtsam::Vector3_
bspline_angular_vel(const gtsam::Rot3_& rot0, const gtsam::Rot3_& rot1, const gtsam::Rot3_& rot2, const gtsam::Rot3_& rot3, const gtsam::Double_& t, const double knot_interval);

/**
 * @brief Calculate global linear velocity of B-spline interpolated trajectory
 *        Sommer et al., "Efficient Derivative Computation for Cumulative B-Splines on Lie Groups", CVPR2020
 * @param trans0        Translation at t0
 * @param trans1        Translation at t1
 * @param trans2        Translation at t2
 * @param trans3        Translation at t3
 * @param t             Normalized time between t1 and t2 in [0, 1]
 * @param knot_interval Real time interval between spline knots
 * @return              Global linear velocity of the interpolated trajectory
 */
gtsam::Vector3_ bspline_linear_vel(
  const gtsam::Vector3_& trans0,
  const gtsam::Vector3_& trans1,
  const gtsam::Vector3_& trans2,
  const gtsam::Vector3_& trans3,
  const gtsam::Double_& t,
  const double knot_interval);

/**
 * @brief Calculate global linear acceleration of B-spline interpolated trajectory
 *        Sommer et al., "Efficient Derivative Computation for Cumulative B-Splines on Lie Groups", CVPR2020
 * @param trans0        Translation at t0
 * @param trans1        Translation at t1
 * @param trans2        Translation at t2
 * @param trans3        Translation at t3
 * @param t             Normalized time between t1 and t2 in [0, 1]
 * @param knot_interval Real time interval between spline knots
 * @return              Global linear acceleration of the interpolated trajectory
 */
gtsam::Vector3_ bspline_linear_acc(
  const gtsam::Vector3_& trans0,
  const gtsam::Vector3_& trans1,
  const gtsam::Vector3_& trans2,
  const gtsam::Vector3_& trans3,
  const gtsam::Double_& t,
  const double knot_interval);

/**
 * @brief Calculate local linear acceleration and angular velocity (IMU measurement) of B-spline interpolated trajectory
 *
 * @param pose0         Pose at t0
 * @param pose1         Pose at t1
 * @param pose2         Pose at t2
 * @param pose3         Pose at t3
 * @param t             Normalized time between t1 and t2 in [0, 1]
 * @param g             Gravity acceleration
 * @param knot_interval Real time interval between spline knots
 * @return              Local linear acceleration and angular velocity
 */
gtsam::Vector6_ bspline_imu(
  const gtsam::Pose3_ pose0,
  const gtsam::Pose3_ pose1,
  const gtsam::Pose3_ pose2,
  const gtsam::Pose3_ pose3,
  const gtsam::Double_& t,
  const double knot_interval,
  const gtsam::Vector3& g);

// Utility functions
inline gtsam::Pose3_ bspline(const gtsam::Key key1, const gtsam::Double_& t) {
  return bspline(gtsam::Pose3_(key1 - 1), gtsam::Pose3_(key1), gtsam::Pose3_(key1 + 1), gtsam::Pose3_(key1 + 2), t);
}

inline gtsam::Pose3_ bspline_se3(const gtsam::Key key1, const gtsam::Double_& t) {
  return bspline_se3(gtsam::Pose3_(key1 - 1), gtsam::Pose3_(key1), gtsam::Pose3_(key1 + 1), gtsam::Pose3_(key1 + 2), t);
}

inline gtsam::Vector6_ bspline_imu(const gtsam::Key key1, const gtsam::Double_& t, const double knot_interval, const gtsam::Vector3& g) {
  return bspline_imu(gtsam::Pose3_(key1 - 1), gtsam::Pose3_(key1), gtsam::Pose3_(key1 + 1), gtsam::Pose3_(key1 + 2), t, knot_interval, g);
}

}  // namespace gtsam_points
