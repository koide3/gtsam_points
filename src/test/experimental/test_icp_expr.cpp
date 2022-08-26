#include <fstream>
#include <iostream>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam_ext/util/read_points.hpp>
#include <gtsam_ext/types/frame_cpu.hpp>
#include <gtsam_ext/factors/expression_icp_factor.hpp>
#include <gtsam_ext/factors/integrated_icp_factor.hpp>
#include <gtsam_ext/optimizers/levenberg_marquardt_ext.hpp>


int main(int argc, char** argv) {
  auto points1 = gtsam_ext::read_points("data/kitti_07_dump/000000/points.bin");
  auto points2 = gtsam_ext::read_points("data/kitti_07_dump/000001/points.bin");

  gtsam_ext::Frame::Ptr frame1(new gtsam_ext::FrameCPU(points1));
  gtsam_ext::Frame::Ptr frame2(new gtsam_ext::FrameCPU(points2));

  gtsam::Values values;
  values.insert(0, gtsam::Pose3::Identity());
  values.insert(1, gtsam::Pose3::Identity());

  gtsam::NonlinearFactorGraph graph;
  graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3::Identity(), gtsam::noiseModel::Isotropic::Precision(6, 1e6)));

  // *** IntegratedICPFactor ***
  // gtsam_ext::IntegratedICPFactor::shared_ptr factor(new gtsam_ext::IntegratedICPFactor(0, 1, frame1, frame2));
  // factor->set_max_corresponding_distance(5.0);
  // factor->set_num_threads(10);
  // graph.add(factor);

  auto noise_model = gtsam::noiseModel::Isotropic::Precision(3, 1.0);
  auto robust_model = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Huber::Create(1.0), noise_model);

  // auto factors = gtsam_ext::create_icp_factors(0, 1, frame1, frame2, robust_model);
  // graph.add(*factors);

  auto icp_factor = gtsam_ext::create_integrated_icp_factor(0, 1, frame1, frame2, robust_model);
  graph.add(icp_factor);

  gtsam_ext::LevenbergMarquardtExtParams lm_params;
  lm_params.callback = [&](const gtsam_ext::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values& values) { std::cout << status.to_string() << std::endl; };
  gtsam_ext::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
  values = optimizer.optimize();

  values.print();

  return 0;
}