// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

/**
 * @file  basic_scan_matching.cpp
 * @brief This example demonstrates how to perform simple frame-to-frame ICP scan matching with gtsam_points.
 */

#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/factors/integrated_icp_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>

int main(int argc, char** argv) {
  // Read target and source point clouds
  const auto target_points = gtsam_points::read_points("data/kitti_00/000000.bin");
  const auto source_points = gtsam_points::read_points("data/kitti_00/000001.bin");

  // Create gtsam_points::PointCloudCPU instances that hold point data
  const auto target_frame = std::make_shared<gtsam_points::PointCloudCPU>(target_points);
  const auto source_frame = std::make_shared<gtsam_points::PointCloudCPU>(source_points);

  // Create GTSAM values and graph
  gtsam::Values values;
  values.insert(0, gtsam::Pose3());  // Target pose initial guess
  values.insert(1, gtsam::Pose3());  // Source pose initial guess

  gtsam::NonlinearFactorGraph graph;

  // Fix the target pose at the origin
  auto prior_factor = gtsam::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gtsam::Pose3(), gtsam::noiseModel::Isotropic::Precision(6, 1e6));
  graph.add(prior_factor);

  // Create an ICP factor between target and source poses
  auto icp_factor = gtsam::make_shared<gtsam_points::IntegratedICPFactor>(0, 1, target_frame, source_frame);
  icp_factor->set_max_correspondence_distance(5.0);
  graph.add(icp_factor);

  // Create LM optimizer
  gtsam_points::LevenbergMarquardtExtParams lm_params;
  lm_params.set_verbose();
  gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);

  // Optimize
  values = optimizer.optimize();

  // Visualization
  auto viewer = guik::LightViewer::instance();
  viewer->update_drawable("target", std::make_shared<glk::PointCloudBuffer>(target_points), guik::FlatRed());
  viewer->update_drawable("source", std::make_shared<glk::PointCloudBuffer>(source_points), guik::FlatGreen());
  viewer->update_drawable(
    "aligned",
    std::make_shared<glk::PointCloudBuffer>(source_points),
    guik::FlatBlue(values.at<gtsam::Pose3>(1).matrix().cast<float>()));
  viewer->spin();

  return 0;
}