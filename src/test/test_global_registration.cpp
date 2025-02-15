#include <random>
#include <gtest/gtest.h>

#include <gtsam_points/ann/kdtree2.hpp>
#include <gtsam_points/ann/kdtreex.hpp>
#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/features/fpfh_estimation.hpp>
#include <gtsam_points/features/normal_estimation.hpp>
#include <gtsam_points/registration/ransac.hpp>

class GlobalRegistrationTest : public testing::Test, public testing::WithParamInterface<std::string> {
  virtual void SetUp() {
    std::mt19937 mt;

    const std::string dataset_path = "data/kitti_00";
    const auto target_raw = gtsam_points::read_points(dataset_path + "/000000.bin");
    const auto source_raw = gtsam_points::read_points(dataset_path + "/000001.bin");

    target = std::make_shared<gtsam_points::PointCloudCPU>(target_raw);
    target = gtsam_points::voxelgrid_sampling(target, 0.5);
    target->add_normals(gtsam_points::estimate_normals(target->points, target->size(), 10));

    source = std::make_shared<gtsam_points::PointCloudCPU>(source_raw);
    source = gtsam_points::voxelgrid_sampling(source, 0.5);
    source->add_normals(gtsam_points::estimate_normals(source->points, source->size(), 10));

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

INSTANTIATE_TEST_SUITE_P(gtsam_points, GlobalRegistrationTest, testing::Values("RANSAC"), [](const auto& info) { return info.param; });

TEST_P(GlobalRegistrationTest, RegistrationTest) {
  std::mt19937 mt;
  const auto result = gtsam_points::estimate_pose_ransac<
    gtsam_points::PointCloud>(*target, *source, target_features, source_features, *target_tree, *target_features_tree, 1.0, 1000, mt);

  std::cout << "inliers=" << result.inlier_rate << " / " << source->size() << std::endl;

  std::cout << "--- T_target_source ---" << std::endl;
  std::cout << result.T_target_source.matrix() << std::endl;

  size_t num_inliers = 0;
  std::vector<Eigen::Vector4d> corr_lines;
  std::vector<Eigen::Vector4d> corr_colors;
  for (size_t i = 0; i < source->size(); i++) {
    const auto& source_f = source_features[i];
    if (source_f.norm() < 1.0) {
      continue;
    }

    size_t target_index;
    double sq_dist;
    target_features_tree->knn_search(source_f.data(), 1, &target_index, &sq_dist);

    const Eigen::Vector4d source_pt = result.T_target_source * source->points[i];
    const double dist = (source_pt - target->points[target_index]).norm();
    const Eigen::Vector4d color = dist < 5.0 ? Eigen::Vector4d(0.0, 1.0, 0.0, 0.5) : Eigen::Vector4d(1.0, 0.0, 0.0, 0.5);
    num_inliers += dist < 5.0 ? 1 : 0;

    corr_lines.emplace_back(source_pt);
    corr_lines.emplace_back(target->points[target_index]);
    corr_colors.emplace_back(color);
    corr_colors.emplace_back(color);
  }

  std::cout << "inliers=" << num_inliers << " / " << source->size() << std::endl;
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}