#include <iostream>
#include <gtsam_ext/util/read_points.hpp>
#include <gtsam_ext/util/edge_plane_extraction.hpp>

#include <gtsam/slam/PriorFactor.h>
#include <gtsam_ext/factors/integrated_loam_factor.hpp>
#include <gtsam_ext/optimizers/levenberg_marquardt_ext.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>

std::vector<Eigen::Vector4d> read_points(const std::string& filename) {
  const auto points_f = gtsam_ext::read_points4(filename);
  std::vector<Eigen::Vector4d> points(points_f.size());

  std::transform(points_f.begin(), points_f.end(), points.begin(), [](const auto& p) { return Eigen::Vector4d(p.x(), p.y(), p.z(), 1.0); });
  return points;
}

int main(int argc, char** argv) {
  const std::string filename0 = "/home/koide/datasets/kitti/dataset/sequences/00/velodyne/000000.bin";
  const std::string filename1 = "/home/koide/datasets/kitti/dataset/sequences/00/velodyne/000001.bin";

  const auto points0 = read_points(filename0);
  const auto points1 = read_points(filename1);

  // Estimate beam projection angles
  // This estimation needs to be done for only the first frame, and can be re-used for successive frames
  const auto scan_lines = gtsam_ext::estimate_scan_lines(points0.data(), points0.size(), 64, 0.2 * M_PI / 180.0);

  // Edge and plane points extraction
  const auto edge_plane_points0 = gtsam_ext::extract_edge_plane_points(scan_lines, points0.data(), points0.size());
  const auto edge_plane_points1 = gtsam_ext::extract_edge_plane_points(scan_lines, points1.data(), points1.size());

  gtsam::Values values;
  values.insert(0, gtsam::Pose3());
  values.insert(1, gtsam::Pose3());

  gtsam::NonlinearFactorGraph graph;
  graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gtsam::Pose3(), gtsam::noiseModel::Isotropic::Precision(6, 1e6));
  // Create point-to-edge, point-to-plane matching-based constraint
  graph.emplace_shared<gtsam_ext::IntegratedLOAMFactor>(0, 1, edge_plane_points0.first, edge_plane_points0.second, edge_plane_points1.first, edge_plane_points1.second);

  // Alternatively, it is also possible to create edge and plane constraints separately
  // graph.emplace_shared<gtsam_ext::IntegratedPointToEdgeFactor>(0, 1, edge_plane_points0.first, edge_plane_points1.first);
  // graph.emplace_shared<gtsam_ext::IntegratedPointToPlaneFactor>(0, 1, edge_plane_points0.second, edge_plane_points1.second);

  auto viewer = guik::LightViewer::instance();
  viewer->update_drawable("points0", std::make_shared<glk::PointCloudBuffer>(points0), guik::FlatRed());
  viewer->update_drawable("points1", std::make_shared<glk::PointCloudBuffer>(points1), guik::FlatGreen());
  viewer->spin_until_click();

  gtsam_ext::LevenbergMarquardtExtParams lm_params;
  lm_params.setMaxIterations(15);
  lm_params.callback = [&](const gtsam_ext::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values& values) {
    std::cout << status.to_string() << std::endl;

    auto drawable = viewer->find_drawable("points1");
    drawable.first->add<Eigen::Matrix4f>("model_matrix", values.at<gtsam::Pose3>(1).matrix().cast<float>());
    viewer->spin_until_click();
  };
  gtsam_ext::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
  values = optimizer.optimize();

  return 0;
}