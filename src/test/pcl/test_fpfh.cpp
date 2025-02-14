#include <random>
#include <gtest/gtest.h>
#include <pcl/features/pfh.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/pfh_tools.h>

#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/features/fpfh_estimation.hpp>
#include <gtsam_points/features/normal_estimation.hpp>

class FPFHTest : public testing::Test, public testing::WithParamInterface<std::string> {
  virtual void SetUp() {
    const std::string dataset_path = "data/kitti_00";
    const auto target_raw = gtsam_points::read_points(dataset_path + "/000000.bin");

    std::mt19937 mt;
    std::uniform_real_distribution<> noise(-0.03, 0.03);

    target = std::make_shared<gtsam_points::PointCloudCPU>(target_raw);
    target = gtsam_points::randomgrid_sampling(target, 1.0, 5000.0 / target->size(), mt);
    target->add_normals(gtsam_points::estimate_normals(target->points, target->size(), 10));

    std::for_each(target->normals, target->normals + target->size(), [&](auto& n) {
      n = (n + Eigen::Vector4d(noise(mt), noise(mt), noise(mt), 0.0)).normalized();
    });
    target_tree = std::make_shared<gtsam_points::KdTree>(target->points, target->size());

    target_pcl = pcl::make_shared<pcl::PointCloud<pcl::PointNormal>>();
    target_pcl->resize(target->size());
    for (int i = 0; i < target->size(); i++) {
      target_pcl->points[i].getVector4fMap() = target->points[i].cast<float>();
      target_pcl->points[i].getNormalVector4fMap() = target->normals[i].cast<float>();
    }
  }

public:
  gtsam_points::PointCloudCPU::Ptr target;
  gtsam_points::NearestNeighborSearch::Ptr target_tree;

  pcl::PointCloud<pcl::PointNormal>::Ptr target_pcl;
};

TEST_F(FPFHTest, LoadCheck) {
  ASSERT_NE(target->size(), 0);
  ASSERT_EQ(target->size(), target_pcl->size());
}

TEST_F(FPFHTest, PairFeaturesTestRandom) {
  std::mt19937 mt;
  std::uniform_real_distribution<> udist(-1.0, 1.0);

  for (int i = 0; i < 1000; i++) {
    const Eigen::Vector4d p1 = Eigen::Vector4d(udist(mt), udist(mt), udist(mt), 1.0);
    const Eigen::Vector4d p2 = Eigen::Vector4d(udist(mt), udist(mt), udist(mt), 1.0);
    const Eigen::Vector4d n1 = Eigen::Vector4d(udist(mt), udist(mt), udist(mt), 0.0).normalized();
    const Eigen::Vector4d n2 = Eigen::Vector4d(udist(mt), udist(mt), udist(mt), 0.0).normalized();

    Eigen::Vector4d f1 = gtsam_points::compute_pair_features(p1, n1, p2, n2);
    Eigen::Vector4f f2;
    pcl::computePairFeatures(p1.cast<float>(), n1.cast<float>(), p2.cast<float>(), n2.cast<float>(), f2[0], f2[1], f2[2], f2[3]);

    EXPECT_NEAR((f1 - f2.cast<double>()).array().abs().maxCoeff(), 0.0, 1e-4) << f1.transpose() << " vs " << f2.transpose();

    Eigen::Vector4d f3 = gtsam_points::compute_pair_features(p2, n2, p1, n1);
    EXPECT_NEAR((f1 - f3).array().abs().maxCoeff(), 0.0, 1e-4) << f1.transpose() << " vs " << f2.transpose();
  }
}

TEST_F(FPFHTest, PairFeaturesTest) {
  for (int i = 0; i < target->size(); i++) {
    std::vector<size_t> neighbors;
    std::vector<double> sq_dists;
    target_tree->radius_search(target->points[i].data(), 5.0, neighbors, sq_dists);

    for (size_t j : neighbors) {
      const auto& p1 = target->points[i];
      const auto& p2 = target->points[j];
      const auto& n1 = target->normals[i];
      const auto& n2 = target->normals[j];

      const Eigen::Vector4d f1 = gtsam_points::compute_pair_features(p1, n1, p2, n2);
      Eigen::Vector4f f2;
      pcl::computePairFeatures(p1.cast<float>(), n1.cast<float>(), p2.cast<float>(), n2.cast<float>(), f2[0], f2[1], f2[2], f2[3]);

      EXPECT_NEAR((f1 - f2.cast<double>()).array().abs().maxCoeff(), 0.0, 1e-4)
        << "i=" << i << " j=" << j << " f1=" << f1.transpose() << " vs " << "f2=" << f2.transpose();
    }
  }
}

INSTANTIATE_TEST_SUITE_P(gtsam_points, FPFHTest, testing::Values("PFH", "FPFH"), [](const auto& info) { return info.param; });

TEST_P(FPFHTest, ExtractionTest) {
  gtsam_points::FPFHEstimationParams params;
  params.search_radius = 3.0;
  params.num_threads = 2;

  std::vector<Eigen::VectorXd> features;
  std::vector<Eigen::VectorXd> features_pcl;

  if (GetParam() == "PFH") {
    const auto pfh = gtsam_points::estimate_pfh(target->points, target->normals, target->size(), *target_tree, params);
    std::copy(pfh.begin(), pfh.end(), std::back_inserter(features));

    pcl::PFHEstimation<pcl::PointNormal, pcl::PointNormal> pfh_est;
    pfh_est.setInputCloud(target_pcl);
    pfh_est.setInputNormals(target_pcl);
    pfh_est.setSearchMethod(pcl::make_shared<pcl::search::KdTree<pcl::PointNormal>>());
    pfh_est.setRadiusSearch(params.search_radius);

    pcl::PointCloud<pcl::PFHSignature125> pfh_pcl;
    pfh_est.compute(pfh_pcl);

    std::transform(pfh_pcl.begin(), pfh_pcl.end(), std::back_inserter(features_pcl), [](const auto& f) {
      return Eigen::Map<const Eigen::VectorXf>(f.histogram, 125).cast<double>();
    });
  } else if (GetParam() == "FPFH") {
    const auto fpfh = gtsam_points::estimate_fpfh(target->points, target->normals, target->size(), *target_tree, params);
    std::copy(fpfh.begin(), fpfh.end(), std::back_inserter(features));

    pcl::FPFHEstimation<pcl::PointNormal, pcl::PointNormal> fpfh_est;
    fpfh_est.setInputCloud(target_pcl);
    fpfh_est.setInputNormals(target_pcl);
    fpfh_est.setSearchMethod(pcl::make_shared<pcl::search::KdTree<pcl::PointNormal>>());
    fpfh_est.setRadiusSearch(params.search_radius);

    pcl::PointCloud<pcl::FPFHSignature33> fpfh_pcl;
    fpfh_est.compute(fpfh_pcl);

    std::transform(fpfh_pcl.begin(), fpfh_pcl.end(), std::back_inserter(features_pcl), [](const auto& f) {
      return Eigen::Map<const Eigen::VectorXf>(f.histogram, 33).cast<double>();
    });
  }

  ASSERT_EQ(features.size(), target->size());
  for (int i = 0; i < features.size(); i++) {
    const Eigen::VectorXd err = features[i] - features_pcl[i];
    EXPECT_NEAR(err.array().abs().maxCoeff(), 0.0, 0.1) << " f1=" << features[i].transpose() << " vs f2=" << features_pcl[i].transpose();
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}