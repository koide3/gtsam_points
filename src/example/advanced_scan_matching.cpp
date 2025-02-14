// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

/**
 * @file  advanced_scan_matching.cpp
 * @brief This example demonstrates how to use iVox structure to efficiently do frame-to-map scan matching.
 */

#include <boost/format.hpp>

#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/features/covariance_estimation.hpp>
#include <gtsam_points/ann/ivox.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: advanced_scan_matching /path/to/your/kitti/00/velodyne" << std::endl;
    return 0;
  }

  const std::string seq_path = argv[1];

  // Mapping parameters
  const int num_threads = 4;
  const double voxel_resolution = 1.0;
  const double randomsampling_rate = 0.1;

  std::mt19937 mt;
  auto viewer = guik::LightViewer::instance();

  // Create iVox
  gtsam_points::iVox::Ptr ivox(new gtsam_points::iVox(voxel_resolution));

  // Estimated sensor pose
  gtsam::Pose3 T_world_lidar;

  for (int i = 0;; i++) {
    // Read points and replace the last element (w) with 1 for homogeneous transformation
    const std::string points_path = (boost::format("%s/%06d.bin") % seq_path % i).str();
    auto points = gtsam_points::read_points4(points_path);
    if (points.empty()) {
      break;
    }
    std::for_each(points.begin(), points.end(), [](Eigen::Vector4f& p) { p.w() = 1.0f; });

    // Create a frame and do random sampling and covariance estimation
    auto frame = std::make_shared<gtsam_points::PointCloudCPU>(points);
    frame = gtsam_points::random_sampling(frame, randomsampling_rate, mt);
    frame->add_covs(gtsam_points::estimate_covariances(frame->points, frame->size(), 10, num_threads));

    // If it is not the first frame, do frame-to-map scan matching
    if (i != 0) {
      gtsam::Values values;
      values.insert(0, gtsam::Pose3());  // Target pose = Map origin
      values.insert(1, T_world_lidar);   // Source pose initial guess = Last sensor pose

      gtsam::NonlinearFactorGraph graph;
      // Fix the target pose
      graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gtsam::Pose3(), gtsam::noiseModel::Isotropic::Precision(6, 1e6));

      // Create an ICP factor between target (iVox) and source (current frame)
      auto icp_factor =
        gtsam::make_shared<gtsam_points::IntegratedGICPFactor_<gtsam_points::iVox, gtsam_points::PointCloud>>(0, 1, ivox, frame, ivox);
      icp_factor->set_num_threads(num_threads);
      graph.add(icp_factor);

      // Optimize
      gtsam_points::LevenbergMarquardtExtParams lm_params;
      lm_params.setMaxIterations(20);
      lm_params.set_verbose();
      gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
      values = optimizer.optimize();

      // Update the current pose
      T_world_lidar = values.at<gtsam::Pose3>(1);

      Eigen::Matrix4d R = Eigen::Matrix4d::Zero();
      R.block<3, 3>(0, 0) = T_world_lidar.rotation().matrix();

      // Transform the current frame into the map frame
      for (int j = 0; j < frame->size(); j++) {
        frame->points[j] = T_world_lidar.matrix() * frame->points[j];
        frame->covs[j] = R * frame->covs[j] * R.transpose();
      }
    }

    // Insert the transformed current frame into iVox
    ivox->insert(*frame);

    // Visualization
    viewer->update_drawable(guik::anon(), glk::Primitives::coordinate_system(), guik::VertexColor(T_world_lidar.matrix().cast<float>()));
    viewer->update_drawable(
      "current",
      std::make_shared<glk::PointCloudBuffer>(points),
      guik::FlatOrange(T_world_lidar.matrix().cast<float>()).add("point_scale", 2.0f));
    viewer->update_drawable("ivox", std::make_shared<glk::PointCloudBuffer>(ivox->voxel_points()), guik::Rainbow());
    if (!viewer->spin_once()) {
      break;
    }
  }

  return 0;
}