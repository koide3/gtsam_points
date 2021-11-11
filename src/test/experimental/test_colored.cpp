#include <iostream>

#include <gtsam/slam/PriorFactor.h>

#include <gtsam_ext/ann/kdtree.hpp>
#include <gtsam_ext/types/frame_cpu.hpp>
#include <gtsam_ext/util/read_points.hpp>
#include <gtsam_ext/factors/integrated_gicp_factor.hpp>
#include <gtsam_ext/factors/integrated_colored_gicp_factor.hpp>
#include <gtsam_ext/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_ext/optimizers/gradient_descent.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>

gtsam_ext::FrameCPU::Ptr load_points(const std::string& filename) {
  auto data = gtsam_ext::read_points4(filename);
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(data.size());
  std::vector<float> intensities(data.size());

  for (int i = 0; i < data.size(); i++) {
    points[i] = data[i].head<3>();
    intensities[i] = data[i][3];
  }

  auto frame = std::make_shared<gtsam_ext::FrameCPU>(points);
  frame->add_intensities(intensities.data(), intensities.size());

  return frame;
}

int main(int argc, char** argv) {
  auto target = load_points("/home/koide/target.bin");
  auto source = load_points("/home/koide/source.bin");
  auto target_gradients = gtsam_ext::IntensityGradients::estimate(target, 10, 50);
  auto source_gradients = gtsam_ext::IntensityGradients::estimate(source, 10, 50);

  std::shared_ptr<gtsam_ext::KdTree> target_tree(new gtsam_ext::KdTree(target->points, target->size()));

  gtsam::Values values;
  values.insert(0, gtsam::Pose3::identity());
  values.insert(1, gtsam::Pose3::identity());

  gtsam::NonlinearFactorGraph graph;
  graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gtsam::Pose3::identity(), gtsam::noiseModel::Isotropic::Precision(6, 1e6));
  graph.emplace_shared<gtsam_ext::IntegratedColoredGICPFactor>(0, 1, target, source, target_tree, target_gradients);
  // graph.emplace_shared<gtsam_ext::IntegratedGICPFactor>(0, 1, target, source, target_tree);

  auto viewer = guik::LightViewer::instance();
  viewer->use_arcball_camera_control();

  auto target_buffer = std::make_shared<glk::PointCloudBuffer>(target->points, target->size());
  target_buffer->add_intensity(glk::COLORMAP::TURBO, target->intensities_storage);
  // target_buffer->add_color(target_gradients->intensity_gradients);
  viewer->update_drawable("target", target_buffer, guik::VertexColor());

  auto source_buffer = std::make_shared<glk::PointCloudBuffer>(source->points, source->size());
  source_buffer->add_intensity(glk::COLORMAP::TURBO, source->intensities_storage);
  viewer->update_drawable("source", source_buffer, guik::VertexColor());

  gtsam_ext::LevenbergMarquardtExtParams lm_params;
  lm_params.callback = [&](const gtsam_ext::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values& values) {
    viewer->append_text(status.to_string());

    gtsam::Pose3 target_pose = values.at<gtsam::Pose3>(0);
    gtsam::Pose3 source_pose = values.at<gtsam::Pose3>(1);

    auto target_drawable = viewer->find_drawable("target");
    target_drawable.first->add("model_matrix", target_pose.matrix().cast<float>().eval());

    auto source_drawable = viewer->find_drawable("source");
    source_drawable.first->add("model_matrix", source_pose.matrix().cast<float>().eval());

    viewer->spin_until_click();
  };

  gtsam_ext::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
  values = optimizer.optimize();

  viewer->spin();

  return 0;
}