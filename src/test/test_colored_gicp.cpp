#include <random>
#include <iostream>

#include <gtest/gtest.h>

#include <gtsam/slam/PriorFactor.h>

#include <gtsam_ext/ann/kdtree.hpp>
#include <gtsam_ext/ann/intensity_kdtree.hpp>
#include <gtsam_ext/types/frame_cpu.hpp>
#include <gtsam_ext/factors/integrated_colored_gicp_factor.hpp>
#include <gtsam_ext/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_ext/util/normal_estimation.hpp>
#include <gtsam_ext/util/covariance_estimation.hpp>

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

    for (double x = -2.0; x <= 2.0; x += 0.02) {
      for (double y = -2.0; y < 2.0; y += 0.02) {
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
    gtsam::Values values;
    values.insert(0, gtsam::Pose3::identity());
    values.insert(1, gtsam::Pose3::identity());

    // Forward test (fix the first)
    gtsam::NonlinearFactorGraph graph;
    graph.add(factor);
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gtsam::Pose3::identity(), gtsam::noiseModel::Isotropic::Precision(6, 1e6));

    values = gtsam_ext::LevenbergMarquardtOptimizerExt(graph, values).optimize();

    Eigen::Isometry3d estimated((values.at<gtsam::Pose3>(0).inverse() * values.at<gtsam::Pose3>(1)).matrix());
    Eigen::Isometry3d error = estimated * delta;
    double error_angle = Eigen::AngleAxisd(error.linear()).angle();
    double error_trans = error.translation().norm();

    EXPECT_LE(error_angle, 0.1 * M_PI / 180.0) << "[FORWARD] Too large rotation error " << tag;
    EXPECT_LE(error_trans, 0.01) << "[FORWARD] Too large translation error" << tag;

    // Backward test (fix the second)
    values.update(0, gtsam::Pose3::identity());
    values.update(1, gtsam::Pose3::identity());

    graph.erase(graph.begin() + 1);
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(1, gtsam::Pose3::identity(), gtsam::noiseModel::Isotropic::Precision(6, 1e6));

    values = gtsam_ext::LevenbergMarquardtOptimizerExt(graph, values).optimize();

    estimated = Eigen::Isometry3d((values.at<gtsam::Pose3>(0).inverse() * values.at<gtsam::Pose3>(1)).matrix());
    error = estimated * delta;
    error_angle = Eigen::AngleAxisd(error.linear()).angle();
    error_trans = error.translation().norm();

    EXPECT_LE(error_angle, 0.1 * M_PI / 180.0) << "[BACKWARD] Too large rotation error" << tag;
    EXPECT_LE(error_trans, 0.01) << "[BACKWARD] Too large translation error" << tag;
  }

  Eigen::Isometry3d delta;
  std::vector<double> target_intensities;
  std::vector<double> source_intensities;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> target_points;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> source_points;
};

TEST_F(ColoredGICPTestBase, Check) {
  gtsam_ext::FrameCPU::Ptr target(new gtsam_ext::FrameCPU(target_points));
  target->add_intensities(target_intensities);
  auto target_gradients = gtsam_ext::IntensityGradients::estimate(target, 10, 50, 1);

  EXPECT_NE(target->normals, nullptr);
  EXPECT_NE(target->covs, nullptr);

  gtsam_ext::Frame::Ptr target_ = target;
  auto target_gradients2 = gtsam_ext::IntensityGradients::estimate(target_, 50, 1);

  gtsam_ext::FrameCPU::Ptr source(new gtsam_ext::FrameCPU(source_points));
  source->add_intensities(source_intensities);
  source->add_covs(gtsam_ext::estimate_covariances(source->points, source->size()));

  std::shared_ptr<gtsam_ext::KdTree> target_tree(new gtsam_ext::KdTree(target->points, target->size()));
  std::shared_ptr<gtsam_ext::IntensityKdTree> target_intensity_tree(new gtsam_ext::IntensityKdTree(target->points, target->intensities, target->size()));

  test_factor(gtsam::make_shared<gtsam_ext::IntegratedColoredGICPFactor>(0, 1, target, source, target_tree, target_gradients), "DEFAULT");
  test_factor(gtsam::make_shared<gtsam_ext::IntegratedColoredGICPFactor>(0, 1, target, source, target_intensity_tree, target_gradients), "ESTIMATE_PHOTO_AND_GEOM");
  test_factor(gtsam::make_shared<gtsam_ext::IntegratedColoredGICPFactor>(0, 1, target, source, target_intensity_tree, target_gradients2), "ESTIMATE_PHOTO_ONLY");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}