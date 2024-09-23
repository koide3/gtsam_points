#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <boost/filesystem.hpp>

#include <gtest/gtest.h>
#include <gtsam_points/config.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/types/point_cloud_gpu.hpp>
#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_points/types/gaussian_voxelmap_gpu.hpp>

#include <random_set.hpp>
#include <compare_frames.hpp>
#include <validate_frame.hpp>

template <typename T, int D>
void creation_test() {
  RandomSet<T, D> randomset;
  const int num_points = randomset.num_points;
  const auto& points = randomset.points;
  const auto& normals = randomset.normals;
  const auto& covs = randomset.covs;
  const auto& intensities = randomset.intensities;
  const auto& times = randomset.times;
  const auto& aux1 = randomset.aux1;
  const auto& aux2 = randomset.aux2;

  auto frame = std::make_shared<gtsam_points::PointCloudCPU>();

  EXPECT_FALSE(frame->has_points());
  frame->add_points(points);

  EXPECT_TRUE(frame->has_points() && frame->check_points());
  compare_frames(frame, std::make_shared<gtsam_points::PointCloudCPU>(points));
  compare_frames(frame, gtsam_points::PointCloudCPU::clone(*frame));

  for (int i = 0; i < num_points; i++) {
    const double diff = (frame->points[i].template head<D>() - points[i].template cast<double>()).squaredNorm();
    EXPECT_LT(diff, std::numeric_limits<double>::epsilon()) << "point copy failure";
    EXPECT_DOUBLE_EQ(frame->points[i][3], 1.0) << "point copy failure";
  }

  EXPECT_FALSE(frame->has_times());
  EXPECT_FALSE(frame->has_normals());
  EXPECT_FALSE(frame->has_covs());
  EXPECT_FALSE(frame->has_intensities());

  frame->add_times(times);
  compare_frames(frame, gtsam_points::PointCloudCPU::clone(*frame));
  EXPECT_TRUE(frame->has_times() && frame->check_times());

  frame->add_covs(covs);
  compare_frames(frame, gtsam_points::PointCloudCPU::clone(*frame));
  EXPECT_TRUE(frame->has_covs() && frame->check_covs());

  frame->add_normals(normals);
  compare_frames(frame, gtsam_points::PointCloudCPU::clone(*frame));
  EXPECT_TRUE(frame->has_normals() && frame->check_normals());

  frame->add_intensities(intensities);
  compare_frames(frame, gtsam_points::PointCloudCPU::clone(*frame));
  EXPECT_TRUE(frame->has_intensities() && frame->check_intensities());

  frame->add_aux_attribute("aux1", aux1);
  compare_frames(frame, gtsam_points::PointCloudCPU::clone(*frame));
  frame->add_aux_attribute("aux2", aux2);
  compare_frames(frame, gtsam_points::PointCloudCPU::clone(*frame));

  for (int i = 0; i < num_points; i++) {
    const double diff_t = std::pow(frame->times[i] - times[i], 2);
    const double diff_n = (frame->normals[i].template head<D>() - normals[i].template cast<double>()).squaredNorm();
    const double diff_c = (frame->covs[i].template block<D, D>(0, 0) - covs[i].template cast<double>()).squaredNorm();
    EXPECT_LT(diff_t, std::numeric_limits<double>::epsilon()) << "time copy failure";
    EXPECT_LT(diff_n, std::numeric_limits<double>::epsilon()) << "normal copy failure";
    EXPECT_LT(diff_c, std::numeric_limits<double>::epsilon()) << "cov copy failure";

    EXPECT_DOUBLE_EQ(frame->normals[i][3], 0.0) << "normal copy failure";
    for (int j = 0; j < 4; j++) {
      EXPECT_DOUBLE_EQ(frame->covs[i](3, j), 0.0) << "cov copy failure";
      EXPECT_DOUBLE_EQ(frame->covs[i](j, 3), 0.0) << "cov copy failure";
    }
  }

  boost::filesystem::create_directories("/tmp/frame_test");
  frame->save("/tmp/frame_test");
  compare_frames(frame, gtsam_points::PointCloudCPU::load("/tmp/frame_test"));

  boost::filesystem::create_directories("/tmp/frame_test_compact");
  frame->save_compact("/tmp/frame_test_compact");
  compare_frames(frame, gtsam_points::PointCloudCPU::load("/tmp/frame_test_compact"));
}

TEST(TestTypes, TestPointCloudCPU) {
  creation_test<float, 3>();
  creation_test<float, 4>();
  creation_test<double, 3>();
  creation_test<double, 4>();
}

#ifdef GTSAM_POINTS_USE_CUDA

template <typename T, int D>
void creation_test_gpu() {
  RandomSet<T, D> randomset;
  const int num_points = randomset.num_points;
  const auto& points = randomset.points;
  const auto& normals = randomset.normals;
  const auto& covs = randomset.covs;
  const auto& intensities = randomset.intensities;
  const auto& times = randomset.times;

  auto frame = std::make_shared<gtsam_points::PointCloudCPU>();
  auto frame_gpu = std::make_shared<gtsam_points::PointCloudGPU>();

  // add_points
  ASSERT_FALSE(frame_gpu->has_points_gpu());
  frame->add_points(points);
  frame_gpu->add_points_gpu(points);
  ASSERT_FALSE(frame_gpu->has_points());
  ASSERT_TRUE(frame_gpu->has_points_gpu() && frame_gpu->check_points_gpu());

  const auto points_gpu = gtsam_points::download_points_gpu(*frame_gpu);
  ASSERT_EQ(points_gpu.size(), num_points);
  for (int i = 0; i < num_points; i++) {
    EXPECT_LT((points_gpu[i].template cast<double>() - frame->points[i].template head<3>()).norm(), 1e-6);
  }

  frame_gpu->download_points();
  compare_frames(frame, frame_gpu);

  frame_gpu->add_points(points);
  compare_frames(frame, frame_gpu);
  compare_frames(frame, gtsam_points::PointCloudGPU::clone(*frame));

  // add_times
  ASSERT_FALSE(frame_gpu->has_times_gpu());
  frame->add_times(times);
  frame_gpu->add_times_gpu(times);
  ASSERT_FALSE(frame_gpu->has_times());
  ASSERT_TRUE(frame_gpu->has_times_gpu() && frame_gpu->check_times_gpu());

  const auto times_gpu = gtsam_points::download_times_gpu(*frame_gpu);
  ASSERT_EQ(times_gpu.size(), num_points);
  for (int i = 0; i < num_points; i++) {
    EXPECT_LT(std::abs(times_gpu[i] - frame->times[i]), 1e-6);
  }

  frame_gpu->add_times(times);
  compare_frames(frame, frame_gpu);
  compare_frames(frame, gtsam_points::PointCloudGPU::clone(*frame));

  // add_intensities
  ASSERT_FALSE(frame_gpu->has_intensities_gpu());
  frame->add_intensities(intensities);
  frame_gpu->add_intensities_gpu(intensities);
  ASSERT_FALSE(frame_gpu->has_intensities());
  ASSERT_TRUE(frame_gpu->has_intensities_gpu() && frame_gpu->check_intensities_gpu());

  const auto intensities_gpu = gtsam_points::download_intensities_gpu(*frame_gpu);
  ASSERT_EQ(intensities_gpu.size(), num_points);
  for (int i = 0; i < num_points; i++) {
    EXPECT_LT(std::abs(intensities_gpu[i] - frame->intensities[i]), 1e-6);
  }

  frame_gpu->add_intensities(intensities);
  compare_frames(frame, frame_gpu);
  compare_frames(frame, gtsam_points::PointCloudGPU::clone(*frame));

  // add_normals
  ASSERT_FALSE(frame_gpu->has_normals_gpu());
  frame->add_normals(normals);
  frame_gpu->add_normals_gpu(normals);
  ASSERT_FALSE(frame_gpu->has_normals());
  ASSERT_TRUE(frame_gpu->has_normals_gpu() && frame_gpu->check_normals_gpu());

  const auto normals_gpu = gtsam_points::download_normals_gpu(*frame_gpu);
  ASSERT_EQ(normals_gpu.size(), num_points);
  for (int i = 0; i < num_points; i++) {
    EXPECT_LT((normals_gpu[i].template cast<double>() - frame->normals[i].template head<3>()).norm(), 1e-6);
  }

  frame_gpu->add_normals(normals);
  compare_frames(frame, frame_gpu);
  compare_frames(frame, gtsam_points::PointCloudGPU::clone(*frame));

  // add_covs
  ASSERT_FALSE(frame_gpu->has_covs_gpu());
  frame->add_covs(covs);
  frame_gpu->add_covs_gpu(covs);
  ASSERT_FALSE(frame_gpu->has_covs());
  ASSERT_TRUE(frame_gpu->has_covs_gpu() && frame_gpu->check_covs_gpu());

  const auto covs_gpu = gtsam_points::download_covs_gpu(*frame_gpu);
  ASSERT_EQ(covs_gpu.size(), num_points);
  for (int i = 0; i < num_points; i++) {
    EXPECT_LT((covs_gpu[i].template cast<double>() - frame->covs[i].template block<3, 3>(0, 0)).norm(), 1e-6);
  }

  frame_gpu->add_covs(covs);
  compare_frames(frame, frame_gpu);
  compare_frames(frame, gtsam_points::PointCloudGPU::clone(*frame));
}

TEST(TestTypes, TestPointCloudGPU) {
  creation_test_gpu<float, 3>();
  creation_test_gpu<float, 4>();
  creation_test_gpu<double, 3>();
  creation_test_gpu<double, 4>();
}

#endif

TEST(TestTypes, TestPointCloudCPUFuncs) {
  RandomSet<double, 4> randomset;
  auto frame = std::make_shared<gtsam_points::PointCloudCPU>();
  frame->add_points(randomset.points);
  frame->add_normals(randomset.normals);
  frame->add_covs(randomset.covs);
  frame->add_intensities(randomset.intensities);
  frame->add_times(randomset.times);
  frame->add_aux_attribute("aux1", randomset.aux1);
  frame->add_aux_attribute("aux2", randomset.aux2);

  // Test for gtsam_points::sample()
  std::mt19937 mt;
  std::vector<int> indices(frame->size());
  std::iota(indices.begin(), indices.end(), 0);

  const int num_samples = frame->size() * 0.5;
  std::vector<int> samples(num_samples);
  std::sample(indices.begin(), indices.end(), samples.begin(), num_samples, mt);

  auto sampled = gtsam_points::sample(frame, samples);
  ASSERT_EQ(sampled->size(), num_samples);
  validate_all_propaties(sampled);

  const Eigen::Vector4d* aux1_ = frame->aux_attribute<Eigen::Vector4d>("aux1");
  const Eigen::Matrix4d* aux2_ = frame->aux_attribute<Eigen::Matrix4d>("aux2");
  const Eigen::Vector4d* aux1 = sampled->aux_attribute<Eigen::Vector4d>("aux1");
  const Eigen::Matrix4d* aux2 = sampled->aux_attribute<Eigen::Matrix4d>("aux2");
  for (int i = 0; i < samples.size(); i++) {
    const int idx = samples[i];
    EXPECT_LT((frame->points[idx] - sampled->points[i]).norm(), 1e-6);
    EXPECT_LT((frame->normals[idx] - sampled->normals[i]).norm(), 1e-6);
    EXPECT_LT((frame->covs[idx] - sampled->covs[i]).norm(), 1e-6);
    EXPECT_DOUBLE_EQ(frame->intensities[idx], sampled->intensities[i]);
    EXPECT_DOUBLE_EQ(frame->times[idx], sampled->times[i]);
    EXPECT_LT((aux1_[idx] - aux1[i]).norm(), 1e-6);
    EXPECT_LT((aux2_[idx] - aux2[i]).norm(), 1e-6);
  }

  // Test for random_sampling, voxelgrid_sampling, and randomgrid_sampling
  sampled = gtsam_points::random_sampling(frame, 0.5, mt);
  EXPECT_DOUBLE_EQ(static_cast<double>(sampled->size()) / frame->size(), 0.5);
  validate_all_propaties(sampled);

  sampled = gtsam_points::voxelgrid_sampling(frame, 0.1);
  EXPECT_LE(sampled->size(), frame->size());
  validate_all_propaties(sampled, false);

  sampled = gtsam_points::randomgrid_sampling(frame, 0.1, 0.5, mt);
  EXPECT_LE(sampled->size(), frame->size());
  validate_all_propaties(sampled);

  // Test for filter
  auto filtered1 = gtsam_points::filter(frame, [](const Eigen::Vector4d& pt) { return pt.x() < 0.0; });
  auto filtered2 = gtsam_points::filter(frame, [](const Eigen::Vector4d& pt) { return pt.x() >= 0.0; });

  validate_all_propaties(filtered1);
  validate_all_propaties(filtered2);
  EXPECT_EQ(filtered1->size() + filtered2->size(), frame->size());
  EXPECT_TRUE(std::all_of(filtered1->points, filtered1->points + filtered1->size(), [](const Eigen::Vector4d& pt) { return pt.x() < 0.0; }));
  EXPECT_TRUE(std::all_of(filtered2->points, filtered2->points + filtered2->size(), [](const Eigen::Vector4d& pt) { return pt.x() >= 0.0; }));

  // Test for filter_by_index
  filtered1 = gtsam_points::filter_by_index(frame, [&](int i) { return frame->points[i].x() < 0.0; });
  filtered2 = gtsam_points::filter_by_index(frame, [&](int i) { return frame->points[i].x() >= 0.0; });

  validate_all_propaties(filtered1);
  validate_all_propaties(filtered2);
  EXPECT_EQ(filtered1->size() + filtered2->size(), frame->size());
  EXPECT_TRUE(std::all_of(filtered1->points, filtered1->points + filtered1->size(), [](const Eigen::Vector4d& pt) { return pt.x() < 0.0; }));
  EXPECT_TRUE(std::all_of(filtered2->points, filtered2->points + filtered2->size(), [](const Eigen::Vector4d& pt) { return pt.x() >= 0.0; }));

  // Test for sort
  auto sorted = gtsam_points::sort(frame, [&](int lhs, int rhs) { return frame->points[lhs].x() < frame->points[rhs].x(); });
  validate_all_propaties(sorted);
  EXPECT_EQ(sorted->size(), frame->size());
  EXPECT_TRUE(std::is_sorted(sorted->points, sorted->points + sorted->size(), [](const auto& lhs, const auto& rhs) { return lhs.x() < rhs.x(); }));

  // Test for sort_by_time
  sorted = gtsam_points::sort_by_time(frame);
  validate_all_propaties(sorted);
  EXPECT_EQ(sorted->size(), frame->size());
  EXPECT_TRUE(std::is_sorted(sorted->times, sorted->times + sorted->size()));

  // Test for transform
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = Eigen::AngleAxisd(0.5, Eigen::Vector3d::Random().normalized()).toRotationMatrix();
  T.translation() = Eigen::Vector3d::Random();

  auto transformed = gtsam_points::transform(frame, T);
  validate_all_propaties(transformed);
  ASSERT_EQ(transformed->size(), frame->size());
  for (int i = 0; i < frame->size(); i++) {
    EXPECT_LT((T * frame->points[i] - transformed->points[i]).norm(), 1e-6);
    EXPECT_LT((T.linear() * frame->normals[i].head<3>() - transformed->normals[i].head<3>()).norm(), 1e-6);
    EXPECT_LT((T.linear() * frame->covs[i].topLeftCorner<3, 3>() * T.linear().transpose() - transformed->covs[i].topLeftCorner<3, 3>()).norm(), 1e-6);
  }

  auto transformed2 = gtsam_points::transform(frame, T.cast<float>());
  compare_frames(transformed, transformed2, "transform<Isometry3f>");

  // Test for transform_inplace
  transformed2 = gtsam_points::PointCloudCPU::clone(*frame);
  gtsam_points::transform_inplace(transformed2, T);
  compare_frames(transformed, transformed2, "transform_inplace<Isometry3d>");

  transformed2 = gtsam_points::PointCloudCPU::clone(*frame);
  gtsam_points::transform_inplace(transformed2, T.cast<float>());
  compare_frames(transformed, transformed2, "transform_inplace<Isometry3f>");

  // Test for remove_outliers
  auto filtered = gtsam_points::remove_outliers(frame);
  EXPECT_LE(filtered->size(), frame->size());
  validate_all_propaties(filtered);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}