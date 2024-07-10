#include <iostream>

#include <gtsam/slam/expressions.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam_points/util/expressions.hpp>
#include <gtsam_points/factors/pose3_interpolation_factor.hpp>

#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>

gtsam::Rot3_ interpolate_(const gtsam::Rot3_& X, const gtsam::Rot3_& Y, const double t) {
  const auto delta = gtsam::logmap(X, Y);
  const auto Delta = gtsam_points::expmap(t * delta);
  const auto result = gtsam::compose(X, Delta);
  return result;
}

gtsam::Vector6_ interpolate_(const gtsam::Pose3_& xi_, const gtsam::Pose3_& xj_, const gtsam::Pose3_& xk_, const double t) {
  const auto Rint_ = interpolate_(gtsam::rotation(xi_), gtsam::rotation(xj_), t);
  const auto Re_ = gtsam::between(gtsam::rotation(xk_), Rint_);
  const auto re_ = gtsam_points::logmap(Re_);

  const auto te_ = gtsam::translation(xk_) - (1.0 - t) * gtsam::translation(xi_) - t * gtsam::translation(xj_);
  const auto error_ = gtsam_points::concatenate(re_, te_);

  return error_;
}

int main(int argc, char** argv) {
  gtsam::Values values;
  values.insert(0, gtsam::Pose3::Expmap(gtsam::Vector6::Random()));
  values.insert(1, gtsam::Pose3::Expmap(gtsam::Vector6::Random()));
  values.insert(2, gtsam::Pose3::Expmap(gtsam::Vector6::Random()));

  /*
  auto noise_model = gtsam::noiseModel::Isotropic::Precision(6, 1.0);
  auto factor = gtsam::make_shared<gtsam_points::Pose3InterpolationFactor>(0, 1, 2, t, noise_model);

  gtsam::Matrix H_xi, H_xj, H_xk;
  auto error = factor->evaluateError(values.at<gtsam::Pose3>(0), values.at<gtsam::Pose3>(1), values.at<gtsam::Pose3>(2), H_xi, H_xj, H_xk);
  */

  /*
  std::cout << "error: " << error.transpose() << std::endl;
  std::cout << "--- H_xi ---" << std::endl << H_xi << std::endl;
  std::cout << "--- H_xj ---" << std::endl << H_xj << std::endl;
  std::cout << "--- H_xk ---" << std::endl << H_xk << std::endl;

  gtsam::Pose3_ xi_(0);
  gtsam::Pose3_ xj_(1);
  gtsam::Pose3_ xk_(2);



  std::vector<gtsam::Matrix> Hs(error_.keys().size());
  const auto error2 = error_.value(values, Hs);

  std::cout << "error:" << error2.transpose() << std::endl;
  std::cout << "--- H_xi ---" << std::endl << Hs[0] << std::endl;
  std::cout << "--- H_xj ---" << std::endl << Hs[1] << std::endl;
  std::cout << "--- H_xk ---" << std::endl << Hs[2] << std::endl;
  */

  auto viewer = guik::LightViewer::instance();
  viewer->update_drawable("xi", glk::Primitives::coordinate_system(), guik::VertexColor(values.at<gtsam::Pose3>(0).matrix().cast<float>()));
  viewer->update_drawable("xj", glk::Primitives::coordinate_system(), guik::VertexColor(values.at<gtsam::Pose3>(1).matrix().cast<float>()));

  float time = 0.0f;
  viewer->register_ui_callback("ui", [&] {
    ImGui::DragFloat("time", &time, 0.01f, 0.0f, 1.0f);

    auto noise_model = gtsam::noiseModel::Isotropic::Precision(6, 1.0);

    gtsam::NonlinearFactorGraph graph;
    graph.emplace_shared<gtsam_points::Pose3InterpolationFactor>(0, 1, 2, time, noise_model);
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, values.at<gtsam::Pose3>(0), noise_model);
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(1, values.at<gtsam::Pose3>(1), noise_model);

    gtsam::LevenbergMarquardtParams lm_params;
    lm_params.setVerbosityLM("SUMMARY");

    std::cout << "factor" << std::endl;
    auto optimized = gtsam::LevenbergMarquardtOptimizer(graph, values, lm_params).optimize();

    std::cout << "expression factor" << std::endl;
    const auto error_ = interpolate_(gtsam::Pose3_(0), gtsam::Pose3_(1), gtsam::Pose3_(2), time);
    // gtsam::ExpressionFactor<gtsam::Vector6>(noise_model, gtsam::Vector6::Zero(), error_);
    graph[0] = gtsam::make_shared<gtsam::ExpressionFactor<gtsam::Vector6>>(noise_model, gtsam::Vector6::Zero(), error_);
    optimized = gtsam::LevenbergMarquardtOptimizer(graph, values, lm_params).optimize();

    const gtsam::Pose3 xk = optimized.at<gtsam::Pose3>(2);
    std::cout << "--- xk ---" << std::endl << xk.matrix() << std::endl;

    viewer->update_drawable("xk", glk::Primitives::coordinate_system(), guik::VertexColor(xk.matrix().cast<float>()));
  });

  viewer->spin();

  return 0;
}