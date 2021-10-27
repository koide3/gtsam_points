#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <boost/filesystem.hpp>

#include <gtest/gtest.h>
#include <gtsam_ext/types/frame_cpu.hpp>
#include <gtsam_ext/types/frame_gpu.hpp>
#include <gtsam_ext/types/voxelized_frame_cpu.hpp>
#include <gtsam_ext/types/voxelized_frame_gpu.hpp>

template <typename T, int D>
struct RandomSet {
  RandomSet() : num_points(128), points(num_points), normals(num_points), covs(num_points), times(num_points) {
    for (int i = 0; i < num_points; i++) {
      points[i].setOnes();
      points[i].template head<3>() = Eigen::Matrix<T, 3, 1>::Random();
      normals[i].setZero();
      normals[i].template head<3>() = Eigen::Matrix<T, 3, 1>::Random();
      covs[i].setZero();
      covs[i].template block<3, 3>(0, 0) = Eigen::Matrix<T, 3, 3>::Random();
      covs[i] = (covs[i] * covs[i].transpose()).eval();
      times[i] = Eigen::Vector2d::Random()[0];
    }
  }

  const int num_points;
  std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>> points;
  std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>> normals;
  std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>> covs;
  std::vector<T> times;
};

void compare_frames(const gtsam_ext::Frame::ConstPtr& frame1, const gtsam_ext::Frame::ConstPtr& frame2) {
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
}

template <typename T, int D>
void creation_test() {
  RandomSet<T, D> randomset;
  const int num_points = randomset.num_points;
  const auto& points = randomset.points;
  const auto& normals = randomset.normals;
  const auto& covs = randomset.covs;
  const auto& times = randomset.times;

  auto frame = std::make_shared<gtsam_ext::FrameCPU>();
  frame->add_points(points);
  compare_frames(frame, std::make_shared<gtsam_ext::FrameCPU>(points));
  compare_frames(frame, std::make_shared<gtsam_ext::FrameCPU>(*frame));

  for (int i = 0; i < num_points; i++) {
    const double diff = (frame->points[i].template head<D>() - points[i].template cast<double>()).squaredNorm();
    EXPECT_LT(diff, std::numeric_limits<double>::epsilon()) << "point copy failure";
    EXPECT_DOUBLE_EQ(frame->points[i][3], 1.0) << "point copy failure";
  }

  EXPECT_EQ(frame->times, nullptr);
  EXPECT_EQ(frame->normals, nullptr);
  EXPECT_EQ(frame->covs, nullptr);

  frame->add_times(times);
  compare_frames(frame, std::make_shared<gtsam_ext::FrameCPU>(*frame));
  frame->add_covs(covs);
  compare_frames(frame, std::make_shared<gtsam_ext::FrameCPU>(*frame));
  frame->add_normals(normals);
  compare_frames(frame, std::make_shared<gtsam_ext::FrameCPU>(*frame));

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
  compare_frames(frame, gtsam_ext::FrameCPU::load("/tmp/frame_test"));

  boost::filesystem::create_directories("/tmp/frame_test_compact");
  frame->save_compact("/tmp/frame_test_compact");
  compare_frames(frame, gtsam_ext::FrameCPU::load("/tmp/frame_test_compact"));
}

TEST(TestTypes, TestFrameCPU) {
  creation_test<float, 3>();
  creation_test<float, 4>();
  creation_test<double, 3>();
  creation_test<double, 4>();
}

template <typename T, int D>
void creation_test_voxels() {
  RandomSet<T, D> randomset;
  const int num_points = randomset.num_points;
  const auto& points = randomset.points;
  const auto& normals = randomset.normals;
  const auto& covs = randomset.covs;
  const auto& times = randomset.times;

  auto frame = std::make_shared<gtsam_ext::FrameCPU>(points);
  frame->add_covs(covs);

  auto voxelized_frame = std::make_shared<gtsam_ext::VoxelizedFrameCPU>();
  voxelized_frame->add_points(points);
  voxelized_frame->add_covs(covs);
  voxelized_frame->create_voxelmap(0.1);
  ASSERT_NE(voxelized_frame->voxels, nullptr);

  compare_frames(frame, voxelized_frame);
  compare_frames(frame, std::make_shared<gtsam_ext::VoxelizedFrameCPU>(0.1, points, covs));
  compare_frames(frame, std::make_shared<gtsam_ext::VoxelizedFrameCPU>(0.1, *frame));

  frame->add_times(times);
  voxelized_frame->add_times(times);
  compare_frames(frame, voxelized_frame);

  frame->add_normals(normals);
  voxelized_frame->add_normals(normals);
  compare_frames(frame, voxelized_frame);

  frame->add_covs(covs);
  voxelized_frame->add_covs(covs);
  compare_frames(frame, voxelized_frame);

  compare_frames(frame, std::make_shared<gtsam_ext::VoxelizedFrameCPU>(0.1, *frame));
}

TEST(TestTypes, TestVoxelizedFrameCPU) {
  creation_test_voxels<float, 3>();
  creation_test_voxels<float, 4>();
  creation_test_voxels<double, 3>();
  creation_test_voxels<double, 4>();
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

  auto frame = std::make_shared<gtsam_ext::FrameCPU>();
  frame->add_points(points);

  auto frame_gpu = std::make_shared<gtsam_ext::FrameGPU>();

  // add_points
  frame_gpu->add_points_gpu(points);
  ASSERT_EQ(frame_gpu->points, nullptr);
  ASSERT_NE(frame_gpu->points_gpu, nullptr);

  const auto points_gpu = frame_gpu->get_points_gpu();
  ASSERT_EQ(points_gpu.size(), num_points);
  for (int i = 0; i < num_points; i++) {
    EXPECT_LT((points_gpu[i].template cast<double>() - frame->points[i].template head<3>()).norm(), 1e-6);
  }

  frame_gpu->add_points(points);
  compare_frames(frame, frame_gpu);
  compare_frames(frame, std::make_shared<gtsam_ext::FrameGPU>(*frame));

  // add_covs
  frame->add_covs(covs);
  frame_gpu->add_covs_gpu(covs);
  ASSERT_EQ(frame_gpu->covs, nullptr);
  ASSERT_NE(frame_gpu->covs_gpu, nullptr);

  const auto covs_gpu = frame_gpu->get_covs_gpu();
  ASSERT_EQ(covs_gpu.size(), num_points);
  for (int i = 0; i < num_points; i++) {
    EXPECT_LT((covs_gpu[i].template cast<double>() - frame->covs[i].template block<3, 3>(0, 0)).norm(), 1e-6);
  }

  frame_gpu->add_covs(covs);
  compare_frames(frame, frame_gpu);
  compare_frames(frame, std::make_shared<gtsam_ext::FrameGPU>(*frame));
}

TEST(TestTypes, TestFrameGPU) {
  creation_test_gpu<float, 3>();
  creation_test_gpu<float, 4>();
  creation_test_gpu<double, 3>();
  creation_test_gpu<double, 4>();
}

template <typename T, int D>
void creation_test_voxels_gpu() {
  RandomSet<T, D> randomset;
  const int num_points = randomset.num_points;
  const auto& points = randomset.points;
  const auto& normals = randomset.normals;
  const auto& covs = randomset.covs;
  const auto& times = randomset.times;

  auto frame = std::make_shared<gtsam_ext::FrameCPU>();
  frame->add_points(points);

  auto frame_gpu = std::make_shared<gtsam_ext::VoxelizedFrameGPU>();

  // add_points
  frame_gpu->add_points_gpu(points);
  ASSERT_EQ(frame_gpu->points, nullptr);
  ASSERT_NE(frame_gpu->points_gpu, nullptr);

  const auto points_gpu = frame_gpu->get_points_gpu();
  ASSERT_EQ(points_gpu.size(), num_points);
  for (int i = 0; i < num_points; i++) {
    EXPECT_LT((points_gpu[i].template cast<double>() - frame->points[i].template head<3>()).norm(), 1e-6);
  }

  frame_gpu->add_points(points);
  compare_frames(frame, frame_gpu);

  // add_covs
  frame->add_covs(covs);
  frame_gpu->add_covs_gpu(covs);
  ASSERT_EQ(frame_gpu->covs, nullptr);
  ASSERT_NE(frame_gpu->covs_gpu, nullptr);

  const auto covs_gpu = frame_gpu->get_covs_gpu();
  ASSERT_EQ(covs_gpu.size(), num_points);
  for (int i = 0; i < num_points; i++) {
    EXPECT_LT((covs_gpu[i].template cast<double>() - frame->covs[i].template block<3, 3>(0, 0)).norm(), 1e-6);
  }

  frame_gpu->add_covs(covs);
  compare_frames(frame, frame_gpu);
  compare_frames(frame, std::make_shared<gtsam_ext::VoxelizedFrameGPU>(0.1, *frame));

  EXPECT_EQ(frame_gpu->voxels_gpu, nullptr);
  frame_gpu->create_voxelmap(0.5);
  EXPECT_NE(frame_gpu->voxels_gpu, nullptr);

  const auto voxel_num_points_gpu = frame_gpu->get_voxel_num_points();
  const auto voxel_means_gpu = frame_gpu->get_voxel_means_gpu();
  const auto voxel_covs_gpu = frame_gpu->get_voxel_covs_gpu();

  auto find_corresponding_voxel = [&](const Eigen::Vector3f& mean_gpu) -> gtsam_ext::GaussianVoxel::Ptr {
    for (const auto& voxel : frame_gpu->voxels->voxels) {
      if ((mean_gpu.template cast<double>() - voxel.second->mean.template head<3>()).norm() < 1e-6) {
        return voxel.second;
      }
    }
    return nullptr;
  };

  EXPECT_EQ(voxel_means_gpu.size(), frame_gpu->voxels->voxels.size());
  for (int i = 0; i < voxel_num_points_gpu.size(); i++) {
    auto voxel_cpu = find_corresponding_voxel(voxel_means_gpu[i]);
    ASSERT_NE(voxel_cpu, nullptr);
    EXPECT_EQ(voxel_cpu->num_points, voxel_num_points_gpu[i]);
    EXPECT_LT((voxel_means_gpu[i].template cast<double>() - voxel_cpu->mean.template head<3>()).norm(), 1e-6);
    EXPECT_LT((voxel_covs_gpu[i].template cast<double>() - voxel_cpu->cov.template block<3, 3>(0, 0)).norm(), 1e-6);
  }
}

TEST(TestTypes, TestVoxelizedFrameGPU) {
  creation_test_voxels_gpu<float, 3>();
  creation_test_voxels_gpu<float, 4>();
  creation_test_voxels_gpu<double, 3>();
  creation_test_voxels_gpu<double, 4>();
}

#endif

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}