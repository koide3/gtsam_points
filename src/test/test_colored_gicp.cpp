#include <random>
#include <iostream>

#include <gtest/gtest.h>

#include <gtsam/slam/PriorFactor.h>

#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/ann/intensity_kdtree.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/factors/integrated_colored_gicp_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/util/parallelism.hpp>
#include <gtsam_points/features/normal_estimation.hpp>
#include <gtsam_points/features/covariance_estimation.hpp>

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

class ColoredGICPTest : public ColoredGICPTestBase, public testing::WithParamInterface<std::tuple<std::string, std::string>> {};

INSTANTIATE_TEST_SUITE_P(
  gtsam_points,
  ColoredGICPTest,
  testing::Combine(testing::Values("GICP"), testing::Values("NONE", "OMP", "TBB")),
  [](const auto& info) { return std::get<0>(info.param) + "_" + std::get<1>(info.param); });

TEST_P(ColoredGICPTest, AlignmentTest) {
  const auto param = GetParam();
  const std::string method = std::get<0>(param);
  const std::string parallelism = std::get<1>(param);
  const int num_threads = parallelism == "NONE" ? 1 : 2;

#ifndef GTSAM_POINTS_USE_TBB
  if (parallelism == "TBB") {
    std::cerr << "Skip test for TBB" << std::endl;
    return;
  }
#endif

  if (parallelism == "TBB") {
    gtsam_points::set_tbb_as_default();
  } else {
    gtsam_points::set_omp_as_default();
  }

  gtsam_points::PointCloudCPU::Ptr target(new gtsam_points::PointCloudCPU(target_points));
  target->add_intensities(target_intensities);
  auto target_gradients = gtsam_points::IntensityGradients::estimate(target, 10, 50, num_threads);

  EXPECT_NE(target->normals, nullptr);
  EXPECT_NE(target->covs, nullptr);

  gtsam_points::PointCloud::Ptr target_ = target;
  auto target_gradients2 = gtsam_points::IntensityGradients::estimate(target_, 50, num_threads);

  gtsam_points::PointCloudCPU::Ptr source(new gtsam_points::PointCloudCPU(source_points));
  source->add_intensities(source_intensities);
  source->add_covs(gtsam_points::estimate_covariances(source->points, source->size()));

  std::shared_ptr<gtsam_points::KdTree> target_tree(new gtsam_points::KdTree(target->points, target->size()));
  std::shared_ptr<gtsam_points::IntensityKdTree> target_intensity_tree(
    new gtsam_points::IntensityKdTree(target->points, target->intensities, target->size()));

  auto f1 = gtsam::make_shared<gtsam_points::IntegratedColoredGICPFactor>(0, 1, target, source, target_tree, target_gradients);
  f1->set_num_threads(num_threads);
  test_factor(f1, "DEFAULT");

  auto f2 = gtsam::make_shared<gtsam_points::IntegratedColoredGICPFactor>(0, 1, target, source, target_intensity_tree, target_gradients);
  f2->set_num_threads(num_threads);
  test_factor(f2, "ESTIMATE_PHOTO_AND_GEOM");

  auto f3 = gtsam::make_shared<gtsam_points::IntegratedColoredGICPFactor>(0, 1, target, source, target_intensity_tree, target_gradients2);
  f3->set_num_threads(num_threads);
  test_factor(f3, "ESTIMATE_PHOTO_ONLY");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}