#include <random>
#include <iostream>
#include <boost/format.hpp>

#include <gtsam_points/ann/ivox.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>

#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/util/edge_plane_extraction.hpp>
#include <gtsam_points/features/covariance_estimation.hpp>

#include <gtsam_points/factors/integrated_loam_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>

gtsam_points::ScanLineInformation estimate_scan_lines() {
  const std::string path = "/home/koide/datasets/kitti/dataset/sequences/00/velodyne/000000.bin";
  const auto points_f = gtsam_points::read_points4(path);
  std::vector<Eigen::Vector4d> points(points_f.size());
  std::transform(points_f.begin(), points_f.end(), points.begin(), [](const auto& p) { return Eigen::Vector4d(p[0], p[1], p[2], 1.0); });

  return gtsam_points::estimate_scan_lines(points.data(), points.size(), 64, 0.2 * M_PI / 180.0);
}

int main(int argc, char** argv) {
  const std::string seq_path = "/home/koide/datasets/kitti/dataset/sequences/00/velodyne";

  auto viewer = guik::LightViewer::instance();

  gtsam_points::iVox::Ptr ivox_edges(new gtsam_points::iVox(1.0, 0.1, 20));
  gtsam_points::iVox::Ptr ivox_planes(new gtsam_points::iVox(1.0, 0.1, 20));

  gtsam_points::ScanLineInformation scan_lines = estimate_scan_lines();

  gtsam::Pose3 T_world_lidar;

  for (int i = 0; i < 4500; i++) {
    const std::string path = (boost::format("%s/%06d.bin") % seq_path % i).str();
    const auto points_f = gtsam_points::read_points4(path);
    std::vector<Eigen::Vector4d> points(points_f.size());
    std::transform(points_f.begin(), points_f.end(), points.begin(), [](const auto& p) { return Eigen::Vector4d(p[0], p[1], p[2], 1.0); });

    auto edges_planes = gtsam_points::extract_edge_plane_points(scan_lines, points.data(), points.size());
    auto edges = edges_planes.first;
    auto planes = edges_planes.second;

    if (i != 0) {
      gtsam::Values values;
      values.insert(0, gtsam::Pose3());
      values.insert(1, T_world_lidar);

      gtsam::NonlinearFactorGraph graph;
      graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gtsam::Pose3(), gtsam::noiseModel::Isotropic::Precision(6, 1e6));

      auto factor = gtsam::make_shared<gtsam_points::IntegratedLOAMFactor_<gtsam_points::iVox, gtsam_points::PointCloud>>(
        0,
        1,
        ivox_edges,
        ivox_planes,
        edges,
        planes,
        ivox_edges,
        ivox_planes);
      factor->set_num_threads(12);
      factor->set_max_correspondence_distance(1.0, 1.0);
      graph.add(factor);

      gtsam_points::LevenbergMarquardtExtParams lm_params;
      lm_params.setMaxIterations(20);
      lm_params.set_verbose();

      gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
      values = optimizer.optimize();

      T_world_lidar = values.at<gtsam::Pose3>(1);
    }

    for (int i = 0; i < edges->size(); i++) {
      edges->points[i] = T_world_lidar.matrix() * edges->points[i];
    }
    for (int i = 0; i < planes->size(); i++) {
      planes->points[i] = T_world_lidar.matrix() * planes->points[i];
    }

    ivox_edges->insert(*edges);
    ivox_planes->insert(*planes);

    const auto plane_points = ivox_planes->voxel_points();
    const auto edge_points = ivox_edges->voxel_points();

    viewer->update_drawable(guik::anon(), glk::Primitives::coordinate_system(), guik::VertexColor(T_world_lidar.matrix().cast<float>()));
    viewer->update_drawable("plane_points", std::make_shared<glk::PointCloudBuffer>(plane_points), guik::Rainbow());
    viewer->update_drawable("edge_points", std::make_shared<glk::PointCloudBuffer>(edge_points), guik::FlatRed());
    viewer->lookat(T_world_lidar.translation().cast<float>());

    viewer->spin_until_click();
  }

  return 0;
}