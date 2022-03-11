#include <gtsam_ext/util/continuous_trajectory.hpp>

#include <vector>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>

#include <gtsam_ext/util/bspline.hpp>

namespace gtsam_ext {

ContinuousTrajectory::ContinuousTrajectory(const char symbol, const double start_time, const double end_time, const double knot_interval, const RotTransMotionModel motion_model)
: symbol(symbol),
  start_time(start_time - knot_interval / 2),
  end_time(end_time + knot_interval / 2),
  knot_interval(knot_interval),
  motion_model(motion_model) {}

const double ContinuousTrajectory::knot_stamp(const int i) const {
  return start_time + (i - 1) * knot_interval;
}

const int ContinuousTrajectory::knot_id(const double t) const {
  return static_cast<int>(std::floor((t - start_time) / knot_interval)) + 1;
}

const int ContinuousTrajectory::knot_max_id() const {
  return knot_id(end_time) + 2;
}

gtsam::Values ContinuousTrajectory::fit_knots(const std::vector<double>& stamps, const std::vector<Eigen::Isometry3d>& poses, double smoothness) const {
  gtsam::Values values;
  gtsam::NonlinearFactorGraph graph;

  // Initial guess estimation
  int pose_cursor = 0;
  for (int i = 0; i <= knot_max_id(); i++) {
    const double t = knot_stamp(i);

    while (pose_cursor < stamps.size() - 2 && std::abs(stamps[pose_cursor] - t) > std::abs(stamps[pose_cursor + 1] - t)) {
      pose_cursor++;
    }

    values.insert(gtsam::Symbol(symbol, i), gtsam::Pose3(poses[pose_cursor].matrix()));
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

    gtsam::Pose3_ pose_ = motion_model == RotTransMotionModel::INDEPENDENT ? bspline(gtsam::Symbol(symbol, knot_i), p) : bspline_se3(gtsam::Symbol(symbol, knot_i), p);

    const auto noise_model = gtsam::noiseModel::Isotropic::Precision(6, 1.0);
    graph.emplace_shared<gtsam::ExpressionFactor<gtsam::Pose3>>(noise_model, gtsam::Pose3(poses[i].matrix()), pose_);
  }

  // Optimize knot poses
  gtsam::LevenbergMarquardtParams lm_params;
  lm_params.setVerbosityLM("SUMMARY");
  gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, lm_params);
  values = optimizer.optimize();

  return values;
}
}  // namespace gtsam_ext