#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <boost/filesystem.hpp>

#include <gtest/gtest.h>
#include <gtsam_ext/types/point_cloud_cpu.hpp>
#include <gtsam_ext/types/point_cloud_gpu.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_gpu.hpp>

template <typename T, int D>
struct RandomSet {
  RandomSet()
  : num_points(128),
    points(num_points),
    normals(num_points),
    covs(num_points),
    intensities(num_points),
    times(num_points),
    aux1(num_points),
    aux2(num_points) {
    //
    for (int i = 0; i < num_points; i++) {
      points[i].setOnes();
      points[i].template head<3>() = Eigen::Matrix<T, 3, 1>::Random();
      normals[i].setZero();
      normals[i].template head<3>() = Eigen::Matrix<T, 3, 1>::Random();
      covs[i].setZero();
      covs[i].template block<3, 3>(0, 0) = Eigen::Matrix<T, 3, 3>::Random();
      covs[i] = (covs[i] * covs[i].transpose()).eval();
      intensities[i] = Eigen::Vector2d::Random()[0];
      times[i] = Eigen::Vector2d::Random()[0];

      aux1[i] = Eigen::Matrix<T, D, 1>::Random();
      aux2[i] = Eigen::Matrix<T, D, D>::Random();
    }
  }

  const int num_points;
  std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>> points;
  std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>> normals;
  std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>> covs;
  std::vector<T> intensities;
  std::vector<T> times;

  std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>> aux1;
  std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>> aux2;
};

void compare_frames(const gtsam_ext::PointCloud::ConstPtr& frame1, const gtsam_ext::PointCloud::ConstPtr& frame2) {
  ASSERT_NE(frame1, nullptr);
  ASSERT_NE(frame2, nullptr);

  EXPECT_EQ(frame1->size(), frame2->size()) << "frame size mismatch";

  if (frame1->points) {
    EXPECT_NE(frame2->points, nullptr);
    for (int i = 0; i < frame1->size(); i++) {
      EXPECT_LT((frame1->points[i] - frame2->points[i]).norm(), 1e-6);
    }
  } else {
    EXPECT_EQ(frame1->points, frame2->points);
  }

  if (frame1->times) {
    EXPECT_NE(frame1->times, nullptr);
    for (int i = 0; i < frame1->size(); i++) {
      EXPECT_LT(abs(frame1->times[i] - frame2->times[i]), 1e-6);
    }
  } else {
    EXPECT_EQ(frame1->times, frame2->times);
  }

  if (frame1->normals) {
    EXPECT_NE(frame1->normals, nullptr);
    for (int i = 0; i < frame1->size(); i++) {
      EXPECT_LT((frame1->normals[i] - frame2->normals[i]).norm(), 1e-6);
    }
  } else {
    EXPECT_EQ(frame1->normals, frame2->normals);
  }

  if (frame1->covs) {
    EXPECT_NE(frame1->covs, nullptr);
    for (int i = 0; i < frame1->size(); i++) {
      EXPECT_LT((frame1->covs[i] - frame2->covs[i]).norm(), 1e-6);
    }
  } else {
    EXPECT_EQ(frame1->covs, frame2->covs);
  }

  if (frame1->intensities) {
    EXPECT_NE(frame1->intensities, nullptr);
    for (int i = 0; i < frame1->size(); i++) {
      EXPECT_LT(abs(frame1->intensities[i] - frame2->intensities[i]), 1e-6);
    }
  } else {
    EXPECT_EQ(frame1->intensities, frame2->intensities);
  }

  for (const auto& aux : frame1->aux_attributes) {
    const auto& name = aux.first;
    const size_t aux_size = aux.second.first;
    const char* aux1_ptr = reinterpret_cast<const char*>(aux.second.second);

    EXPECT_TRUE(frame2->aux_attributes.count(name));
    const auto found = frame2->aux_attributes.find(name);
    if (found == frame2->aux_attributes.end()) {
      continue;
    }

    EXPECT_EQ(found->second.first, aux_size);
    if (found->second.first != aux_size) {
      continue;
    }

    const char* aux2_ptr = reinterpret_cast<const char*>(found->second.second);
    EXPECT_TRUE(std::equal(aux1_ptr, aux1_ptr + aux_size * frame1->size(), aux2_ptr));
  }
}

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

  auto frame = std::make_shared<gtsam_ext::PointCloudCPU>();

  EXPECT_FALSE(frame->has_points());
  frame->add_points(points);

  EXPECT_TRUE(frame->has_points() && frame->check_points());
  compare_frames(frame, std::make_shared<gtsam_ext::PointCloudCPU>(points));
  compare_frames(frame, std::make_shared<gtsam_ext::PointCloudCPU>(*frame));

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
  compare_frames(frame, std::make_shared<gtsam_ext::PointCloudCPU>(*frame));
  EXPECT_TRUE(frame->has_times() && frame->check_times());

  frame->add_covs(covs);
  compare_frames(frame, std::make_shared<gtsam_ext::PointCloudCPU>(*frame));
  EXPECT_TRUE(frame->has_covs() && frame->check_covs());

  frame->add_normals(normals);
  compare_frames(frame, std::make_shared<gtsam_ext::PointCloudCPU>(*frame));
  EXPECT_TRUE(frame->has_normals() && frame->check_normals());

  frame->add_intensities(intensities);
  compare_frames(frame, std::make_shared<gtsam_ext::PointCloudCPU>(*frame));
  EXPECT_TRUE(frame->has_intensities() && frame->check_intensities());

  frame->add_aux_attribute("aux1", aux1);
  compare_frames(frame, std::make_shared<gtsam_ext::PointCloudCPU>(*frame));
  frame->add_aux_attribute("aux2", aux2);
  compare_frames(frame, std::make_shared<gtsam_ext::PointCloudCPU>(*frame));

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
  compare_frames(frame, gtsam_ext::PointCloudCPU::load("/tmp/frame_test"));

  boost::filesystem::create_directories("/tmp/frame_test_compact");
  frame->save_compact("/tmp/frame_test_compact");
  compare_frames(frame, gtsam_ext::PointCloudCPU::load("/tmp/frame_test_compact"));
}

TEST(TestTypes, TestPointCloudCPU) {
  creation_test<float, 3>();
  creation_test<float, 4>();
  creation_test<double, 3>();
  creation_test<double, 4>();
}

#ifdef BUILD_GTSAM_EXT_GPU

template <typename T, int D>
void creation_test_gpu() {
  RandomSet<T, D> randomset;
  const int num_points = randomset.num_points;
  const auto& points = randomset.points;
  const auto& normals = randomset.normals;
  const auto& covs = randomset.covs;
  const auto& times = randomset.times;

  auto frame = std::make_shared<gtsam_ext::PointCloudCPU>();
  frame->add_points(points);

  auto frame_gpu = std::make_shared<gtsam_ext::PointCloudGPU>();

  // add_points
  frame_gpu->add_points_gpu(points);
  ASSERT_EQ(frame_gpu->points, nullptr);
  ASSERT_NE(frame_gpu->points_gpu, nullptr);

  const auto points_gpu = gtsam_ext::download_points_gpu(*frame_gpu);
  ASSERT_EQ(points_gpu.size(), num_points);
  for (int i = 0; i < num_points; i++) {
    EXPECT_LT((points_gpu[i].template cast<double>() - frame->points[i].template head<3>()).norm(), 1e-6);
  }

  frame_gpu->add_points(points);
  compare_frames(frame, frame_gpu);
  compare_frames(frame, std::make_shared<gtsam_ext::PointCloudGPU>(*frame));

  // add_covs
  frame->add_covs(covs);
  frame_gpu->add_covs_gpu(covs);
  ASSERT_EQ(frame_gpu->covs, nullptr);
  ASSERT_NE(frame_gpu->covs_gpu, nullptr);

  const auto covs_gpu = gtsam_ext::download_covs_gpu(*frame_gpu);
  ASSERT_EQ(covs_gpu.size(), num_points);
  for (int i = 0; i < num_points; i++) {
    EXPECT_LT((covs_gpu[i].template cast<double>() - frame->covs[i].template block<3, 3>(0, 0)).norm(), 1e-6);
  }

  frame_gpu->add_covs(covs);
  compare_frames(frame, frame_gpu);
  compare_frames(frame, std::make_shared<gtsam_ext::PointCloudGPU>(*frame));
}

TEST(TestTypes, TestPointCloudGPU) {
  creation_test_gpu<float, 3>();
  creation_test_gpu<float, 4>();
  creation_test_gpu<double, 3>();
  creation_test_gpu<double, 4>();
}

#endif

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}