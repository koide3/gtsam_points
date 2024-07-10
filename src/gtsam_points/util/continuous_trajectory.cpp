// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/util/continuous_trajectory.hpp>

#include <vector>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>

#include <gtsam_points/util/bspline.hpp>

namespace gtsam_points {

ContinuousTrajectory::ContinuousTrajectory(char symbol, double start_time, double end_time, double knot_interval)
: symbol(symbol),
  start_time(start_time - knot_interval / 2),
  end_time(end_time + knot_interval / 2),
  knot_interval(knot_interval) {}

double ContinuousTrajectory::knot_stamp(int i) const {
  return start_time + (i - 1) * knot_interval;
}

int ContinuousTrajectory::knot_id(double t) const {
  return static_cast<int>(std::floor((t - start_time) / knot_interval)) + 1;
}

int ContinuousTrajectory::knot_max_id() const {
  return knot_id(end_time) + 2;
}

gtsam::Pose3_ ContinuousTrajectory::pose(double t, const gtsam::Double_& t_) {
  const int knot_i = knot_id(t);
  const double knot_t = knot_stamp(knot_i);
  const gtsam::Double_ p_ = (1.0 / knot_interval) * (t_ - gtsam::Double_(knot_t));

  return gtsam_points::bspline(gtsam::Symbol(symbol, knot_i), p_);
}

gtsam::Pose3 ContinuousTrajectory::pose(const gtsam::Values& values, double t) {
  const auto pose_ = pose(t, gtsam::Double_(t));
  return pose_.value(values);
}

gtsam::Vector6_ ContinuousTrajectory::imu(double t, const gtsam::Double_& t_, const Eigen::Vector3d& g) {
  const int knot_i = knot_id(t);
  const double knot_t = knot_stamp(knot_i);
  const gtsam::Double_ p_ = (1.0 / knot_interval) * (t_ - gtsam::Double_(knot_t));

  return gtsam_points::bspline_imu(gtsam::Symbol(symbol, knot_i), p_, knot_interval, g);
}

gtsam::Vector6 ContinuousTrajectory::imu(const gtsam::Values& values, double t, const Eigen::Vector3d& g) {
  const auto imu_ = imu(t, gtsam::Double_(t), g);
  return imu_.value(values);
}

gtsam::Values
ContinuousTrajectory::fit_knots(const std::vector<double>& stamps, const std::vector<gtsam::Pose3>& poses, double smoothness, bool verbose) const {
  gtsam::LevenbergMarquardtParams lm_params;
  if (verbose) {
    lm_params.setVerbosityLM("SUMMARY");
  }
  return fit_knots(stamps, poses, smoothness, lm_params);
}

gtsam::Values ContinuousTrajectory::fit_knots(
  const std::vector<double>& stamps,
  const std::vector<gtsam::Pose3>& poses,
  double smoothness,
  const gtsam::LevenbergMarquardtParams& lm_params) const {
  gtsam::Values values;
  gtsam::NonlinearFactorGraph graph;

  // Initial guess estimation
  int pose_cursor = 0;
  for (int i = 0; i <= knot_max_id(); i++) {
    const double t = knot_stamp(i);

    while (pose_cursor < stamps.size() - 2 && std::abs(stamps[pose_cursor] - t) > std::abs(stamps[pose_cursor + 1] - t)) {
      pose_cursor++;
    }

    values.insert(gtsam::Symbol(symbol, i), poses[pose_cursor]);
  }

  // Smoothness regularization to prevent optimization corruption
  if (smoothness > 0.0) {
    const auto noise_model = gtsam::noiseModel::Isotropic::Precision(6, smoothness);
    for (int i = 0; i < knot_max_id(); i++) {
      graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(gtsam::Symbol(symbol, i), gtsam::Symbol(symbol, i + 1), gtsam::Pose3(), noise_model);
    }
  }

  // Create fitting constraints
  for (int i = 0; i < stamps.size(); i++) {
    const int knot_i = knot_id(stamps[i]);
    const double knot_t = knot_stamp(knot_i);
    const double p = (stamps[i] - knot_t) / knot_interval;

    gtsam::Pose3_ pose_ = bspline(gtsam::Symbol(symbol, knot_i), p);

    const auto noise_model = gtsam::noiseModel::Isotropic::Precision(6, 1.0);
    graph.emplace_shared<gtsam::ExpressionFactor<gtsam::Pose3>>(noise_model, gtsam::Pose3(poses[i].matrix()), pose_);
  }

  // Optimize knot poses
  gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, lm_params);
  values = optimizer.optimize();

  return values;
}
}  // namespace gtsam_points