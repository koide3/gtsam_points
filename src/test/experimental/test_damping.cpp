#include <iostream>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>

#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam_points/factors/loose_prior_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>

#include <gtsam_points/factors/linear_damping_factor.hpp>

int main(int argc, char** argv) {
  gtsam::Values values;
  values.insert(0, gtsam::Pose3(gtsam::Rot3::identity(), gtsam::Vector3(100.0, 0.0, 0.0)));

  gtsam::NonlinearFactorGraph graph;
  graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gtsam::Pose3(), gtsam::noiseModel::Isotropic::Precision(6, 10.0));
  graph.emplace_shared<gtsam_points::LoosePriorFactor<gtsam::Pose3>>(0, gtsam::Pose3(), gtsam::noiseModel::Isotropic::Precision(6, 10.0));

  gtsam_points::LevenbergMarquardtExtParams lm_params;
  lm_params.callback = [](const gtsam_points::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values& values) {
    std::cout << status.iterations << ":" << values.at<gtsam::Pose3>(0).translation().transpose() << std::endl;
  };
  gtsam_points::LevenbergMarquardtOptimizerExt(graph, values, lm_params).optimize();


  gtsam::Matrix6 G = 10.0 * gtsam::Matrix6::Identity();
  gtsam::Vector6 g = gtsam::Vector6::Zero();
  // graph.at(1) = gtsam::make_shared<gtsam::LinearContainerFactor>(gtsam::HessianFactor(0, G, g, 0.0));

  graph.at(1) = gtsam::make_shared<gtsam_points::LinearDampingFactor>(0, 6, 10.0);

  gtsam_points::LevenbergMarquardtOptimizerExt(graph, values, lm_params).optimize();
  


  return 0;
}