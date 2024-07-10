// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

/**
 * @file  advanced_frame_manipulation.cpp
 * @brief This example code demonstrates how to feed a custom point cloud class to scan matching factors in gtsam_points.
 */

#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/types/frame_traits.hpp>
#include <gtsam_points/factors/integrated_icp_factor.hpp>
#include <gtsam_points/factors/impl/integrated_icp_factor_impl.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>

namespace gtsam_points {
namespace frame {

// In this example, we show how to directly feed a custom class (in this case, std::vector<Eigen::Vector4d>) to the ICP factor.
// You need to first define methods to access the point data in your custom class by specializing gtsam_points::frame::traits<>
template <>
struct traits<std::vector<Eigen::Vector4d>> {
  using T = std::vector<Eigen::Vector4d>;

  // To use the conventional ICP, you need to at least define the following methods:
  // - size(const T& frame)            : Get the number of points (This is required for only source fame)
  // - has_points(const T& frame)      : Check if your custom class has point data
  // - point(const T& frame, size_t i) : Get i-th point from your class
  static int size(const T& points) { return points.size(); }
  static bool has_points(const T& points) { return !points.empty(); }
  static const Eigen::Vector4d& point(const T& points, size_t i) { return points[i]; }
};

}  // namespace frame
}  // namespace gtsam_points

int main(int argc, char** argv) {
  // Read target and source point clouds and transform them into std::shared_ptr<std::vector<Eigen::Vector4d>>
  const auto target_f = gtsam_points::read_points("data/kitti_00/000000.bin");
  std::shared_ptr<std::vector<Eigen::Vector4d>> target(new std::vector<Eigen::Vector4d>(target_f.size()));
  std::transform(target_f.begin(), target_f.end(), target->begin(), [](const Eigen::Vector3f& p) { return Eigen::Vector4d(p.x(), p.y(), p.z(), 1.0); });

  const auto source_f = gtsam_points::read_points("data/kitti_00/000001.bin");
  std::shared_ptr<std::vector<Eigen::Vector4d>> source(new std::vector<Eigen::Vector4d>(source_f.size()));
  std::transform(source_f.begin(), source_f.end(), source->begin(), [](const Eigen::Vector3f& p) { return Eigen::Vector4d(p.x(), p.y(), p.z(), 1.0); });

  // Create KdTree for the target points
  std::shared_ptr<gtsam_points::KdTree> target_tree(new gtsam_points::KdTree(target->data(), target->size()));

  gtsam::NonlinearFactorGraph graph;

  // Fix the target pose
  auto prior_factor = gtsam::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gtsam::Pose3(), gtsam::noiseModel::Isotropic::Precision(6, 1e6));
  graph.add(prior_factor);

  // Create an ICP factor with std::shared_ptr<std::vector<Eigen::Vector4d>>
  // Note that you need to include "integrated_icp_factor_impl.hpp" when you feed a custom data to scan matching factors
  auto icp_factor = gtsam::make_shared<gtsam_points::IntegratedICPFactor_<std::vector<Eigen::Vector4d>, std::vector<Eigen::Vector4d>>>(0, 1, target, source, target_tree);
  graph.add(icp_factor);

  gtsam::Values values;
  values.insert(0, gtsam::Pose3());   // Target pose initial guess
  values.insert(1, gtsam::Pose3());   // Source pose initial guess

  // Create LM optimizer
  gtsam_points::LevenbergMarquardtExtParams lm_params;
  lm_params.set_verbose();
  gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);

  // Optimize
  values = optimizer.optimize();

  // Visualization
  auto viewer = guik::LightViewer::instance();
  viewer->update_drawable("target", std::make_shared<glk::PointCloudBuffer>(*target), guik::FlatRed());
  viewer->update_drawable("source", std::make_shared<glk::PointCloudBuffer>(*source), guik::FlatGreen());
  viewer->update_drawable("aligned", std::make_shared<glk::PointCloudBuffer>(*source), guik::FlatBlue(values.at<gtsam::Pose3>(1).matrix().cast<float>()));
  viewer->spin();

  return 0;
}