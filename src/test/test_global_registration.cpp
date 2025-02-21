#include <random>
#include <gtest/gtest.h>

#include <gtsam_points/ann/kdtree2.hpp>
#include <gtsam_points/ann/kdtreex.hpp>
#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/factors/integrated_icp_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/features/fpfh_estimation.hpp>
#include <gtsam_points/features/normal_estimation.hpp>
#include <gtsam_points/registration/ransac.hpp>
#include <gtsam_points/registration/graduated_non_convexity.hpp>

class GlobalRegistrationTest : public testing::Test, public testing::WithParamInterface<std::tuple<std::string, int>> {
  virtual void SetUp() {
    const std::string dataset_path = "data/kitti_00";
    const auto target_raw = gtsam_points::read_points(dataset_path + "/000000.bin");
    const auto source_raw = gtsam_points::read_points(dataset_path + "/000001.bin");

    target = std::make_shared<gtsam_points::PointCloudCPU>(target_raw);
    target = gtsam_points::voxelgrid_sampling(target, 0.5);
    target->add_normals(gtsam_points::estimate_normals(target->points, target->size(), 10));

    source = std::make_shared<gtsam_points::PointCloudCPU>(source_raw);
    source = gtsam_points::voxelgrid_sampling(source, 0.5);
    source->add_normals(gtsam_points::estimate_normals(source->points, source->size(), 10));

    // Align source to target for ground truth
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;
    graph.emplace_shared<gtsam_points::IntegratedICPFactor>(gtsam::Pose3(), 0, target, source);
    values.insert(0, gtsam::Pose3());
    values = gtsam_points::LevenbergMarquardtOptimizerExt(graph, values).optimize();
    gtsam_points::transform_inplace(source, Eigen::Isometry3d(values.at<gtsam::Pose3>(0).matrix()));

    // Add some rotation and translation
    T_target_source.setIdentity();
    T_target_source.translation() << 20.0, 5.0, 1.0;
    T_target_source.linear() = Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d(0.01, 0.0, 1.0).normalized()).toRotationMatrix();
    gtsam_points::transform_inplace(source, T_target_source.inverse());

    target_tree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(target);
    source_tree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(source);

    gtsam_points::FPFHEstimationParams fpfh_params;
    fpfh_params.search_radius = 5.0;
    target_features = gtsam_points::estimate_fpfh(target->points, target->normals, target->size(), *target_tree, fpfh_params);
    source_features = gtsam_points::estimate_fpfh(source->points, source->normals, source->size(), *source_tree, fpfh_params);

    target_features_tree = std::make_shared<gtsam_points::KdTreeX<gtsam_points::FPFH_DIM>>(target_features.data(), target_features.size());
    source_features_tree = std::make_shared<gtsam_points::KdTreeX<gtsam_points::FPFH_DIM>>(source_features.data(), source_features.size());
  }

public:
  Eigen::Isometry3d T_target_source;
  gtsam_points::PointCloudCPU::Ptr target;
  gtsam_points::PointCloudCPU::Ptr source;
  gtsam_points::NearestNeighborSearch::Ptr target_tree;
  gtsam_points::NearestNeighborSearch::Ptr source_tree;

  std::vector<gtsam_points::FPFHSignature> target_features;
  std::vector<gtsam_points::FPFHSignature> source_features;
  gtsam_points::NearestNeighborSearch::Ptr target_features_tree;
  gtsam_points::NearestNeighborSearch::Ptr source_features_tree;
};

TEST_F(GlobalRegistrationTest, LoadCheck) {
  ASSERT_NE(target->size(), 0);
  ASSERT_NE(source->size(), 0);
  ASSERT_EQ(target->size(), target_features.size());
  ASSERT_EQ(source->size(), source_features.size());
}

INSTANTIATE_TEST_SUITE_P(
  gtsam_points,
  GlobalRegistrationTest,
  testing::Combine(testing::Values("RANSAC", "GNC"), testing::Values(4, 6)),
  [](const auto& info) { return std::get<0>(info.param) + "_" + std::to_string(std::get<1>(info.param)) + "DoF"; });

TEST_P(GlobalRegistrationTest, RegistrationTest) {
  gtsam_points::RegistrationResult result;

  const std::string method = std::get<0>(GetParam());
  const int dof = std::get<1>(GetParam());

  if (method == "RANSAC") {
    gtsam_points::RANSACParams params;
    params.num_threads = 2;
    params.dof = dof;
    result = gtsam_points::estimate_pose_ransac(
      *target,
      *source,
      target_features.data(),
      source_features.data(),
      *target_tree,
      *target_features_tree,
      params);
  } else {
    gtsam_points::GNCParams params;
    params.num_threads = 2;
    params.dof = dof;
    result = gtsam_points::estimate_pose_gnc(
      *target,
      *source,
      target_features.data(),
      source_features.data(),
      *target_tree,
      *target_features_tree,
      *source_features_tree,
      params);
  }

  const Eigen::Isometry3d error = T_target_source.inverse() * result.T_target_source;
  const double error_t = error.translation().norm();
  const double error_r = Eigen::AngleAxisd(error.linear()).angle();
  EXPECT_LE(error_t, 0.5);
  EXPECT_LE(error_r, 0.1);

  if (dof == 4) {
    const Eigen::Vector3d z = result.T_target_source.linear().col(2);
    EXPECT_NEAR((z - Eigen::Vector3d::UnitZ()).cwiseAbs().maxCoeff(), 0.0, 1e-6);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}