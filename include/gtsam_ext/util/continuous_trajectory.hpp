// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <vector>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/expressions.h>

namespace gtsam {
class LevenbergMarquardtParams;
}

namespace gtsam_ext {

/**
 * @brief Continuous trajectory class for offline batch optimization
 *
 */
class ContinuousTrajectory {
public:
  /**
   * @brief Construct a continuous trajectory instance
   * @param symbol        Key symbol
   * @param start_time    Start time of the trajectory
   * @param end_time      End time of the trajectory
   * @param knot_interval Time interval of spline control knots
   */
  ContinuousTrajectory(const char symbol, const double start_time, const double end_time, const double knot_interval);

  /**
   * @brief Time of a spline knot
   */
  const double knot_stamp(const int i) const;

  /**
   * @brief Key knot ID for a given time
   */
  const int knot_id(const double t) const;

  /**
   * @brief Number of spline knots
   */
  const int knot_max_id() const;

  /**
   * @brief Get an expression of the interpolated time at t
   * @param t  Time
   * @param t_ Expression of t
   */
  const gtsam::Pose3_ pose(const double t, const gtsam::Double_& t_);

  /**
   * @brief Calculate the interpolated time at t
   * @param values  Values including knot poses
   * @param t       Time
   */
  const gtsam::Pose3 pose(const gtsam::Values& values, const double t);

  /**
   * @brief Get an expression of the linear acceleration and angular velocity at t
   * @param t  Time
   * @param t_ Expression of t
   */
  const gtsam::Vector6_ imu(const double t, const gtsam::Double_& t_, const Eigen::Vector3d& g = Eigen::Vector3d(0.0, 0.0, 9.80665));

  /**
   * @brief Calculate the linear acceleration and angular velocity at t
   * @param values  Values including knot poses
   * @param t       Time
   */
  const gtsam::Vector6 imu(const gtsam::Values& values, const double t, const Eigen::Vector3d& g = Eigen::Vector3d(0.0, 0.0, 9.80665));

  /**
   * @brief Optimize spline knots to fit the interpolated trajectory to a set of poses
   * @param stamps      Timestamps of target poses
   * @param poses       Target poses
   * @param smoothness  Smoothness regularization to prevent underconstrained system (If smoothness <= 0.0, regularization will be disabled)
   * @return            Knots of B-spline fitted to the target poses
   */
  gtsam::Values fit_knots(
    const std::vector<double>& stamps,
    const std::vector<gtsam::Pose3>& poses,
    double smoothness,
    const gtsam::LevenbergMarquardtParams& lm_params) const;

  gtsam::Values fit_knots(const std::vector<double>& stamps, const std::vector<gtsam::Pose3>& poses, double smoothness = 1e-3, bool verbose = false)
    const;

public:
  const char symbol;
  const double start_time;
  const double end_time;
  const double knot_interval;
};

}  // namespace gtsam_ext
