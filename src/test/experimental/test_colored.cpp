#include <random>
#include <chrono>
#include <iostream>

#include <gtsam/slam/PriorFactor.h>

#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/ann/intensity_kdtree.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/factors/integrated_colored_gicp_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/optimizers/gradient_descent.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>

gtsam_points::PointCloudCPU::Ptr load_points(const std::string& filename) {
  auto data = gtsam_points::read_points4(filename);
  std::vector<Eigen::Vector3f> points(data.size());
  std::vector<float> intensities(data.size());

  for (int i = 0; i < data.size(); i++) {
    points[i] = data[i].head<3>();
    intensities[i] = data[i][3];
  }

  auto frame = std::make_shared<gtsam_points::PointCloudCPU>(points);
  frame->add_intensities(intensities.data(), intensities.size());

  return frame;
}

int main(int argc, char** argv) {
  auto viewer = guik::LightViewer::instance();
  viewer->use_arcball_camera_control();

  std::mt19937 mt;
  std::normal_distribution<> ndist(0.0, 1e-2);

  std::vector<Eigen::Vector4d> target_points;
  std::vector<double> target_intensities;

  for (double x = -4.0; x <= 4.0; x += 0.02) {
    for (double y = -4.0; y < 4.0; y += 0.02) {
      const Eigen::Vector4d pt(x, y, 2.0, 1.0);
      const Eigen::Vector4d noise(ndist(mt), ndist(mt), ndist(mt), 0.0);

      target_points.push_back(pt + noise);

      const int dx = round(x);
      const int dy = round(y);
      const double d = Eigen::Vector2d(x - dx, y - dy).norm();
      const double intensity = d < 0.1 ? 1.0 : 0.0;
      target_intensities.push_back(intensity + ndist(mt));
    }
  }

  Eigen::Isometry3d delta = Eigen::Isometry3d::Identity();
  delta.translation() += Eigen::Vector3d(0.15, 0.15, 0.25);
  delta.linear() = Eigen::AngleAxisd(0.05, Eigen::Vector3d(0.1, 0.2, 0.3).normalized()).toRotationMatrix();

  std::vector<Eigen::Vector4d> source_points;
  std::vector<double> source_intensities;
  for (double x = -2.0; x <= 2.0; x += 0.02) {
    for (double y = -2.0; y < 2.0; y += 0.02) {
      const Eigen::Vector4d pt(x, y, 2.0, 1.0);
      const Eigen::Vector4d noise(ndist(mt), ndist(mt), ndist(mt), 0.0);

      source_points.push_back(delta * pt + noise);

      const int dx = round(x);
      const int dy = round(y);
      const double d = Eigen::Vector2d(x - dx, y - dy).norm();
      const double intensity = d < 0.1 ? 1.0 : 0.0;
      source_intensities.push_back(intensity + ndist(mt));
    }
  }

  auto target_buffer = std::make_shared<glk::PointCloudBuffer>(target_points);
  target_buffer->add_intensity(glk::COLORMAP::TURBO, target_intensities);
  viewer->update_drawable("target", target_buffer, guik::VertexColor());

  auto source_buffer = std::make_shared<glk::PointCloudBuffer>(source_points);
  source_buffer->add_intensity(glk::COLORMAP::TURBO, source_intensities);
  viewer->update_drawable("source", source_buffer, guik::VertexColor());

  auto target = std::make_shared<gtsam_points::PointCloudCPU>(target_points);
  target->add_intensities(target_intensities);
  auto source = std::make_shared<gtsam_points::PointCloudCPU>(source_points);
  source->add_intensities(source_intensities);

  auto target_gradients = gtsam_points::IntensityGradients::estimate(target, 10, 50);
  auto source_gradients = gtsam_points::IntensityGradients::estimate(source, 10, 50);
  // std::shared_ptr<gtsam_points::KdTree> target_tree(new gtsam_points::KdTree(target->points, target->size()));
  std::shared_ptr<gtsam_points::IntensityKdTree> target_tree(new gtsam_points::IntensityKdTree(target->points, target->intensities, target->size()));

  gtsam::Values values;
  values.insert(0, gtsam::Pose3::Identity());
  values.insert(1, gtsam::Pose3::Identity());

  gtsam::NonlinearFactorGraph graph;
  graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gtsam::Pose3::Identity(), gtsam::noiseModel::Isotropic::Precision(6, 1e6));

  auto f = gtsam::make_shared<gtsam_points::IntegratedColoredGICPFactor>(0, 1, target, source, target_tree, target_gradients);
  f->set_photometric_term_weight(0.75);
  // auto f = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(0, 1, target, source);
  f->set_num_threads(12);
  graph.add(f);

  gtsam_points::LevenbergMarquardtExtParams lm_params;
  lm_params.callback = [&](const gtsam_points::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values& values) {
    viewer->append_text(status.to_string());

    gtsam::Pose3 target_pose = values.at<gtsam::Pose3>(0);
    gtsam::Pose3 source_pose = values.at<gtsam::Pose3>(1);

    auto target_drawable = viewer->find_drawable("target");
    target_drawable.first->add("model_matrix", target_pose.matrix().cast<float>().eval());

    auto source_drawable = viewer->find_drawable("source");
    source_drawable.first->add("model_matrix", source_pose.matrix().cast<float>().eval());

    viewer->spin_until_click();
  };

  gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
  values = optimizer.optimize();

  viewer->spin();

  return 0;

  /*
  auto target = load_points("/home/koide/target.bin");
  auto source = load_points("/home/koide/source.bin");
  auto target_gradients = gtsam_points::IntensityGradients::estimate(target, 10, 50);
  auto source_gradients = gtsam_points::IntensityGradients::estimate(source, 10, 50);

  std::shared_ptr<gtsam_points::KdTree> target_tree(new gtsam_points::KdTree(target->points, target->size()));

  gtsam::Values values;
  values.insert(0, gtsam::Pose3::Identity());
  values.insert(1, gtsam::Pose3::Identity());

  gtsam::NonlinearFactorGraph graph;
  graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gtsam::Pose3::Identity(), gtsam::noiseModel::Isotropic::Precision(6, 1e6));

  auto f = gtsam::make_shared<gtsam_points::IntegratedColoredGICPFactor>(0, 1, target, source, target_tree, target_gradients);
  f->set_num_threads(12);
  graph.add(f);
  // graph.emplace_shared<gtsam_points::IntegratedGICPFactor>(0, 1, target, source, target_tree);

  auto target_buffer = std::make_shared<glk::PointCloudBuffer>(target->points, target->size());
  target_buffer->add_intensity(glk::COLORMAP::TURBO, target->intensities_storage);
  // target_buffer->add_color(target_gradients->intensity_gradients);
  viewer->update_drawable("target", target_buffer, guik::VertexColor());

  auto source_buffer = std::make_shared<glk::PointCloudBuffer>(source->points, source->size());
  source_buffer->add_intensity(glk::COLORMAP::TURBO, source->intensities_storage);
  viewer->update_drawable("source", source_buffer, guik::VertexColor());

  gtsam_points::LevenbergMarquardtExtParams lm_params;
  lm_params.callback = [&](const gtsam_points::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values& values) {
    viewer->append_text(status.to_string());

    gtsam::Pose3 target_pose = values.at<gtsam::Pose3>(0);
    gtsam::Pose3 source_pose = values.at<gtsam::Pose3>(1);

    auto target_drawable = viewer->find_drawable("target");
    target_drawable.first->add("model_matrix", target_pose.matrix().cast<float>().eval());

    auto source_drawable = viewer->find_drawable("source");
    source_drawable.first->add("model_matrix", source_pose.matrix().cast<float>().eval());

    viewer->spin_once();
  };

  gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
  values = optimizer.optimize();

  viewer->spin();
  */

  return 0;
}