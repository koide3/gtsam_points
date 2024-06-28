#include <random>
#include <iostream>

#include <gtest/gtest.h>

#include <gtsam/slam/PriorFactor.h>

#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/ann/intensity_kdtree.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/factors/integrated_icp_factor.hpp>
#include <gtsam_points/factors/integrated_color_consistency_factor.hpp>
#include <gtsam_points/factors/integrated_colored_gicp_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/util/normal_estimation.hpp>
#include <gtsam_points/util/covariance_estimation.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>

/*
struct ColoredGICPTestBase : public testing::Test {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  virtual void SetUp() override {
    std::mt19937 mt;
    std::normal_distribution<> ndist(0.0, 1e-2);

    delta = Eigen::Isometry3d::Identity();
    delta.translation() += Eigen::Vector3d(0.15, 0.15, 0.25);
    delta.linear() = Eigen::AngleAxisd(0.05, Eigen::Vector3d(0.1, 0.2, 0.3).normalized()).toRotationMatrix();

    for (double x = -4.0; x <= 4.0; x += 0.02) {
      for (double y = -4.0; y < 4.0; y += 0.02) {
        const Eigen::Vector4d pt(x, y, 2.0, 1.0);

        target_points.push_back(pt + Eigen::Vector4d(ndist(mt), ndist(mt), ndist(mt), 0.0));

        const int dx = round(x);
        const int dy = round(y);
        const double d = Eigen::Vector2d(x - dx, y - dy).norm();
        const double intensity = d < 0.1 ? 1.0 : 0.0;
        target_intensities.push_back(intensity + ndist(mt));
      }
    }

    for (double x = -2.0; x <= 2.0; x += 0.04) {
      for (double y = -2.0; y < 2.0; y += 0.04) {
        const Eigen::Vector4d pt(x, y, 2.0, 1.0);

        source_points.push_back(delta * pt + Eigen::Vector4d(ndist(mt), ndist(mt), ndist(mt), 0.0));

        const int dx = round(x);
        const int dy = round(y);
        const double d = Eigen::Vector2d(x - dx, y - dy).norm();
        const double intensity = d < 0.1 ? 1.0 : 0.0;
        source_intensities.push_back(intensity + ndist(mt));
      }
    }
  }

  void test_factor(const gtsam::NonlinearFactor::shared_ptr& factor, const std::string& tag) {
    const double error_angle_tol = 0.5 * M_PI / 180.0;
    const double error_trans_tol = 0.02;

    gtsam::Values values;
    values.insert(0, gtsam::Pose3::Identity());
    values.insert(1, gtsam::Pose3::Identity());

    // Forward test (fix the first)
    gtsam::NonlinearFactorGraph graph;
    graph.add(factor);
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gtsam::Pose3::Identity(), gtsam::noiseModel::Isotropic::Precision(6, 1e6));

    gtsam_points::LevenbergMarquardtExtParams lm_params;
    lm_params.setMaxIterations(30);
    values = gtsam_points::LevenbergMarquardtOptimizerExt(graph, values, lm_params).optimize();

    Eigen::Isometry3d estimated((values.at<gtsam::Pose3>(0).inverse() * values.at<gtsam::Pose3>(1)).matrix());
    Eigen::Isometry3d error = estimated * delta;
    double error_angle = Eigen::AngleAxisd(error.linear()).angle();
    double error_trans = error.translation().norm();

    EXPECT_LE(error_angle, error_angle_tol) << "[FORWARD] Too large rotation error " << tag;
    EXPECT_LE(error_trans, error_trans_tol) << "[FORWARD] Too large translation error" << tag;

    // Backward test (fix the second)
    values.update(0, gtsam::Pose3::Identity());
    values.update(1, gtsam::Pose3::Identity());

    graph.erase(graph.begin() + 1);
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(1, gtsam::Pose3::Identity(), gtsam::noiseModel::Isotropic::Precision(6, 1e6));

    values = gtsam_points::LevenbergMarquardtOptimizerExt(graph, values, lm_params).optimize();

    estimated = Eigen::Isometry3d((values.at<gtsam::Pose3>(0).inverse() * values.at<gtsam::Pose3>(1)).matrix());
    error = estimated * delta;
    error_angle = Eigen::AngleAxisd(error.linear()).angle();
    error_trans = error.translation().norm();

    EXPECT_LE(error_angle, error_angle_tol) << "[BACKWARD] Too large rotation error" << tag;
    EXPECT_LE(error_trans, error_trans_tol) << "[BACKWARD] Too large translation error" << tag;
  }

  Eigen::Isometry3d delta;
  std::vector<double> target_intensities;
  std::vector<double> source_intensities;
  std::vector<Eigen::Vector4d> target_points;
  std::vector<Eigen::Vector4d> source_points;
};

TEST_F(ColoredGICPTestBase, Check) {
  gtsam_points::PointCloudCPU::Ptr target(new gtsam_points::PointCloudCPU(target_points));
  target->add_intensities(target_intensities);
  auto target_gradients = gtsam_points::IntensityGradients::estimate(target, 10, 50, 1);

  EXPECT_NE(target->normals, nullptr);
  EXPECT_NE(target->covs, nullptr);

  gtsam_points::PointCloud::Ptr target_ = target;
  auto target_gradients2 = gtsam_points::IntensityGradients::estimate(target_, 50, 1);

  gtsam_points::PointCloudCPU::Ptr source(new gtsam_points::PointCloudCPU(source_points));
  source->add_intensities(source_intensities);
  source->add_covs(gtsam_points::estimate_covariances(source->points, source->size()));

  std::shared_ptr<gtsam_points::KdTree> target_tree(new gtsam_points::KdTree(target->points, target->size()));
  std::shared_ptr<gtsam_points::IntensityKdTree> target_intensity_tree(
    new gtsam_points::IntensityKdTree(target->points, target->intensities, target->size()));

  test_factor(gtsam::make_shared<gtsam_points::IntegratedColoredGICPFactor>(0, 1, target, source, target_tree, target_gradients), "DEFAULT");
  test_factor(
    gtsam::make_shared<gtsam_points::IntegratedColoredGICPFactor>(0, 1, target, source, target_intensity_tree, target_gradients),
    "ESTIMATE_PHOTO_AND_GEOM");
  test_factor(
    gtsam::make_shared<gtsam_points::IntegratedColoredGICPFactor>(0, 1, target, source, target_intensity_tree, target_gradients2),
    "ESTIMATE_PHOTO_ONLY");
}
*/

int main(int argc, char** argv) {
  std::mt19937 mt;

  Eigen::Isometry3d delta = Eigen::Isometry3d::Identity();

  gtsam_points::PointCloudCPU::Ptr target;
  gtsam_points::NearestNeighborSearch::Ptr target_tree;
  gtsam_points::IntensityGradients::Ptr target_gradients;

  gtsam_points::PointCloudCPU::Ptr source;

  int method = 0;
  float intensity_scale = 100.0f;

  auto viewer = guik::viewer();
  viewer->disable_xy_grid();

  viewer->register_ui_callback("ui", [&] {
    ImGui::DragFloat("intensity_scale", &intensity_scale);

    if (ImGui::Button("generate points") || target == nullptr) {
      std::normal_distribution<> ndist(0.0, 1e-2);
      std::uniform_real_distribution<> udist;

      delta.translation() += Eigen::Vector3d(0.25 * udist(mt), 0.25 * udist(mt), 0.5 * udist(mt));
      delta.linear() = Eigen::AngleAxisd(0.2 * udist(mt), Eigen::Vector3d(udist(mt), udist(mt), udist(mt)).normalized()).toRotationMatrix();

      std::vector<double> target_intensities;
      std::vector<double> source_intensities;
      std::vector<Eigen::Vector4d> target_points;
      std::vector<Eigen::Vector4d> source_points;
      std::vector<Eigen::Vector4d> target_colors;

      for (double x = -4.0; x <= 4.0; x += 0.02) {
        for (double y = -4.0; y < 4.0; y += 0.02) {
          const Eigen::Vector4d pt(x, y, 0.0, 1.0);

          target_points.push_back(pt + Eigen::Vector4d(ndist(mt), ndist(mt), ndist(mt), 0.0));

          const int dx = round(x);
          const int dy = round(y);
          const double d = Eigen::Vector2d(x - dx, y - dy).norm();
          const double intensity = d < 0.1 ? intensity_scale : 0.0;
          target_intensities.push_back(intensity + intensity_scale * ndist(mt));

          target_colors.push_back(glk::colormapf(glk::COLORMAP::TURBO, target_intensities.back()).cast<double>());
          target_colors.back().w() = 0.5;
        }
      }

      for (double x = -2.0; x <= 2.0; x += 0.04) {
        for (double y = -2.0; y < 2.0; y += 0.04) {
          const Eigen::Vector4d pt(x, y, 0.0, 1.0);

          source_points.push_back(delta * pt + Eigen::Vector4d(ndist(mt), ndist(mt), ndist(mt), 0.0));

          const int dx = round(x);
          const int dy = round(y);
          const double d = Eigen::Vector2d(x - dx, y - dy).norm();
          const double intensity = d < 0.1 ? intensity_scale : 0.0;
          source_intensities.push_back(intensity + intensity_scale * ndist(mt));
        }
      }

      target = std::make_shared<gtsam_points::PointCloudCPU>(target_points);
      target->add_intensities(target_intensities);

      target_tree = std::make_shared<gtsam_points::KdTree>(target->points, target->size());
      target_gradients = gtsam_points::IntensityGradients::estimate(target, 10, 50);

      source = std::make_shared<gtsam_points::PointCloudCPU>(source_points);
      source->add_intensities(source_intensities);
      source->add_covs(gtsam_points::estimate_covariances(source->points, source->size()));

      viewer->update_points("target", target_points, guik::VertexColor().make_transparent())
        ->add_intensity(glk::COLORMAP::TURBO, target_intensities, 1.0 / intensity_scale);
      viewer->update_coord("target_coord", guik::VertexColor());
      viewer->update_points("source", source_points, guik::VertexColor().set_point_scale(3.0f))
        ->add_intensity(glk::COLORMAP::JET, source_intensities, 1.0 / intensity_scale);
      viewer->update_coord("source_coord", guik::VertexColor(delta));
    }

    const std::vector<const char*> methods = {"ColoredICP", "ColoredGICP", "ICP (w/o color)"};
    ImGui::Combo("method", &method, methods.data(), methods.size());

    if (ImGui::Button("align")) {
      gtsam::Values values;
      values.insert(0, gtsam::Pose3());
      values.insert(1, gtsam::Pose3());

      gtsam::NonlinearFactorGraph graph;
      graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gtsam::Pose3(), gtsam::noiseModel::Isotropic::Precision(6, 1e3));

      switch (method) {
        case 0:
          graph.emplace_shared<gtsam_points::IntegratedPointToPlaneICPFactor>(0, 1, target, source, target_tree);
          graph.emplace_shared<gtsam_points::IntegratedColorConsistencyFactor>(0, 1, target, source, target_tree, target_gradients);
          break;

        case 1:
          graph.emplace_shared<gtsam_points::IntegratedColoredGICPFactor>(0, 1, target, source, target_tree, target_gradients);
          break;

        case 2:
          graph.emplace_shared<gtsam_points::IntegratedPointToPlaneICPFactor>(0, 1, target, source, target_tree);
          break;
      }

      gtsam_points::LevenbergMarquardtExtParams lm_params;
      lm_params.setMaxIterations(100);
      lm_params.status_msg_callback = [&](const std::string& msg) { viewer->append_text(msg); };
      lm_params.callback = [&](const gtsam_points::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values& values) {
        viewer->find_drawable("source").first->set_model_matrix(values.at<gtsam::Pose3>(1).matrix());
        viewer->find_drawable("source_coord").first->set_model_matrix(delta.matrix() * values.at<gtsam::Pose3>(1).matrix());
        viewer->spin_once();
      };

      values = gtsam_points::LevenbergMarquardtOptimizerExt(graph, values, lm_params).optimize();
    }
  });

  viewer->spin();

  return 0;
}