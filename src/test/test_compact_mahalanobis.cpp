#include <vector>
#include <iostream>
#include <Eigen/Core>

#include <gtest/gtest.h>
#include <gtsam/base/make_shared.h>
#include <gtsam_points/util/compact.hpp>
#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/features/covariance_estimation.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor.hpp>

TEST(CompactTest, CompactUncompact) {
  for (int i = 0; i < 10; i++) {
    const Eigen::Matrix3d rand3 = Eigen::Matrix3d::Random();
    Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
    cov.topLeftCorner<3, 3>() = rand3 * rand3.transpose();

    const Eigen::Matrix<float, 6, 1> compact = gtsam_points::compact_cov(cov);
    const Eigen::Matrix4d cov2 = gtsam_points::uncompact_cov(compact);

    EXPECT_NEAR((cov - cov2).norm(), 0.0, 1e-6);
  }
}

TEST(CompactTest, UnusedElements) {
  for (int i = 0; i < 10; i++) {
    Eigen::Matrix4d cov = Eigen::Matrix4d::Random();
    cov = cov * cov.transpose();

    const Eigen::Matrix<float, 6, 1> compact = gtsam_points::compact_cov(cov);
    const Eigen::Matrix4d cov2 = gtsam_points::uncompact_cov(compact);

    EXPECT_NEAR(cov2.bottomRows<1>().norm(), 0.0, 1e-6);
    EXPECT_NEAR(cov2.rightCols<1>().norm(), 0.0, 1e-6);
  }
}

class CompactMahalanobisTestBase : public testing::Test {
public:
  virtual void SetUp() override {
    const std::string data_path = "./data/kitti_00";

    const auto load_points = [](const std::string& filename) -> gtsam_points::PointCloudCPU::Ptr {
      const auto raw = gtsam_points::read_points(filename);
      if (raw.empty()) {
        std::cerr << "Failed to read " << filename << std::endl;
        return nullptr;
      }

      auto points = std::make_shared<gtsam_points::PointCloudCPU>(raw);
      points = gtsam_points::voxelgrid_sampling(points, 0.25);
      points->add_covs(gtsam_points::estimate_covariances(points->points, points->size()));
      return points;
    };

    target = load_points(data_path + "/000000.bin");
    source = load_points(data_path + "/000001.bin");
    if (target) {
      target_tree = std::make_shared<gtsam_points::KdTree>(target->points, target->size());

      auto voxels = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(1.0);
      voxels->insert(*target);
      target_voxels = voxels;
    }
  }

  gtsam_points::PointCloudCPU::ConstPtr target;
  gtsam_points::PointCloudCPU::ConstPtr source;
  gtsam_points::KdTree::ConstPtr target_tree;
  gtsam_points::GaussianVoxelMap::ConstPtr target_voxels;
};

TEST_F(CompactMahalanobisTestBase, LoadCheck) {
  ASSERT_NE(target, nullptr);
  ASSERT_NE(source, nullptr);
  ASSERT_NE(target_tree, nullptr);
}

class CompactMahalanobisTest : public CompactMahalanobisTestBase, public testing::WithParamInterface<std::string> {
public:
  std::tuple<gtsam::NonlinearFactor::shared_ptr, gtsam::NonlinearFactor::shared_ptr, gtsam::NonlinearFactor::shared_ptr> create_factor() {
    const auto method = GetParam();

    if (method == "GICP") {
      auto factor_full = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(0, 1, target, source, target_tree);
      factor_full->set_fused_cov_cache_mode(gtsam_points::FusedCovCacheMode::FULL);

      auto factor_compact = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(0, 1, target, source, target_tree);
      factor_compact->set_fused_cov_cache_mode(gtsam_points::FusedCovCacheMode::COMPACT);

      auto factor_none = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(0, 1, target, source, target_tree);
      factor_none->set_fused_cov_cache_mode(gtsam_points::FusedCovCacheMode::NONE);

      return {factor_full, factor_compact, factor_none};
    } else if (method == "VGICP") {
      auto factor_full = gtsam::make_shared<gtsam_points::IntegratedVGICPFactor>(0, 1, target_voxels, source);
      factor_full->set_fused_cov_cache_mode(gtsam_points::FusedCovCacheMode::FULL);

      auto factor_compact = gtsam::make_shared<gtsam_points::IntegratedVGICPFactor>(0, 1, target_voxels, source);
      factor_compact->set_fused_cov_cache_mode(gtsam_points::FusedCovCacheMode::COMPACT);

      auto factor_none = gtsam::make_shared<gtsam_points::IntegratedVGICPFactor>(0, 1, target_voxels, source);
      factor_none->set_fused_cov_cache_mode(gtsam_points::FusedCovCacheMode::NONE);

      return {factor_full, factor_compact, factor_none};
    } else {
      std::cerr << "Unknown method: " << method << std::endl;
    }

    return std::tuple<gtsam::NonlinearFactor::shared_ptr, gtsam::NonlinearFactor::shared_ptr, gtsam::NonlinearFactor::shared_ptr>();
  }
};

INSTANTIATE_TEST_SUITE_P(gtsam_points, CompactMahalanobisTest, testing::Values("GICP", "VGICP"), [](const auto& info) { return info.param; });

TEST_P(CompactMahalanobisTest, FactorTest) {
  const auto factors = create_factor();
  const auto factor_full = std::get<0>(factors);
  const auto factor_compact = std::get<1>(factors);
  const auto factor_none = std::get<2>(factors);

  std::mt19937 mt;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  for (int i = 0; i < 3; i++) {
    gtsam::Vector6 noise1, noise2;
    std::generate(noise1.data(), noise1.data() + noise1.size(), [&] { return dist(mt) * 0.5; });
    std::generate(noise2.data(), noise2.data() + noise2.size(), [&] { return dist(mt) * 0.5; });

    gtsam::Values values;
    values.insert(0, gtsam::Pose3::Expmap(noise1));
    values.insert(1, gtsam::Pose3::Expmap(noise2));

    const auto linear_full = factor_full->linearize(values);
    const auto linear_compact = factor_compact->linearize(values);
    const auto linear_none = factor_none->linearize(values);

    const auto info_full = linear_full->augmentedInformation();
    const auto info_compact = linear_compact->augmentedInformation();
    const auto info_none = linear_none->augmentedInformation();

    EXPECT_NEAR((info_full - info_compact).cwiseAbs2().maxCoeff(), 0.0, 1e-3) << "Large augmented info error (full vs. compact)";
    EXPECT_NEAR((info_full - info_none).cwiseAbs2().maxCoeff(), 0.0, 1e-3) << "Large augmented info error (full vs. none)";

    gtsam::Vector6 noise3;
    std::generate(noise3.data(), noise3.data() + noise3.size(), [&] { return dist(mt) * 0.1; });
    values.insert_or_assign(0, gtsam::Pose3::Expmap(noise2) * gtsam::Pose3::Expmap(noise3));

    const double error_full = factor_full->error(values);
    const double error_compact = factor_compact->error(values);
    const double error_none = factor_none->error(values);

    EXPECT_NEAR(error_full, error_compact, 1e-3) << "Large error (full vs. compact)";
    EXPECT_NEAR(error_full, error_none, 1e-3) << "Large error (full vs. none)";
  }
}
