#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <boost/format.hpp>

#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam_ext/util/read_points.hpp>
#include <gtsam_ext/util/covariance_estimation.hpp>
#include <gtsam_ext/util/easy_profiler.hpp>
#include <gtsam_ext/types/point_cloud_cpu.hpp>
#include <gtsam_ext/types/point_cloud_gpu.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_gpu.hpp>

#include <validate_frame.hpp>

struct VoxelMapTestBase : public testing::Test {
  virtual void SetUp() {
    std::string dump_path = "./data/kitti_07_dump";
    std::ifstream ifs(dump_path + "/graph.txt");
    EXPECT_EQ(ifs.is_open(), true) << "Failed to open " << dump_path;

    // It seems generated random numbers change depending on the compiler
    // Should we use pregenerated randoms for reproductivity?
    const double pose_noise_scale = 0.1;
    std::mt19937 mt(8192 - 1);
    std::uniform_real_distribution<> udist(-pose_noise_scale, pose_noise_scale);

    // Read submap poses
    for (int i = 0; i < 5; i++) {
      std::string token;
      Eigen::Vector3d trans;
      Eigen::Quaterniond quat;
      ifs >> token >> trans.x() >> trans.y() >> trans.z() >> quat.x() >> quat.y() >> quat.z() >> quat.w();

      Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
      pose.translation() = trans;
      pose.linear() = quat.toRotationMatrix();
      poses_gt.insert(i, gtsam::Pose3(pose.matrix()));

      gtsam::Vector6 tan_noise;
      tan_noise << udist(mt), udist(mt), udist(mt), udist(mt), udist(mt), udist(mt);
      poses.insert(i, poses_gt.at<gtsam::Pose3>(i) * gtsam::Pose3::Expmap(tan_noise));
    }

    // Read submap points
    for (int i = 0; i < 5; i++) {
      const std::string points_path = (boost::format("%s/%06d/points.bin") % dump_path % i).str();
      auto points_f = gtsam_ext::read_points(points_path);
      EXPECT_NE(points_f.empty(), true) << "Failed to read points";

      auto frame = std::make_shared<gtsam_ext::PointCloudCPU>(points_f);
      frame = gtsam_ext::randomgrid_sampling(frame, 1.0, 10000.0 / frame->size(), mt);
      frame->add_covs(gtsam_ext::estimate_covariances(frame->points, frame->size()));
      frames.push_back(frame);

#ifdef BUILD_GTSAM_EXT_GPU
      frames.back() = gtsam_ext::PointCloudGPU::clone(*frames.back());
#endif

      auto voxelmap = std::make_shared<gtsam_ext::GaussianVoxelMapCPU>(1.0);
      voxelmap->insert(*frames.back());
      voxelmaps.push_back(voxelmap);

#ifdef BUILD_GTSAM_EXT_GPU
      auto voxelmap_gpu = std::make_shared<gtsam_ext::GaussianVoxelMapGPU>(1.0);
      voxelmap_gpu->insert(*frames.back());
      voxelmaps_gpu.push_back(voxelmap_gpu);
#else
      voxelmaps_gpu.push_back(nullptr);
#endif
    }
  }

  std::vector<gtsam_ext::PointCloud::Ptr> frames;
  std::vector<gtsam_ext::GaussianVoxelMap::Ptr> voxelmaps;
  std::vector<gtsam_ext::GaussianVoxelMap::Ptr> voxelmaps_gpu;
  gtsam::Values poses;
  gtsam::Values poses_gt;
};

TEST_F(VoxelMapTestBase, LoadCheck) {
  ASSERT_EQ(poses.size(), 5) << "Failed to load submap poses";
  ASSERT_EQ(poses_gt.size(), 5) << "Failed to load submap poses";
}

TEST_F(VoxelMapTestBase, VoxelMapCPU) {
  for (int i = 0; i < frames.size(); i++) {
    EXPECT_GT(gtsam_ext::overlap(voxelmaps[i], frames[i], Eigen::Isometry3d::Identity()), 0.99);
  }

  std::vector<gtsam_ext::GaussianVoxelMap::ConstPtr> voxelmaps_(voxelmaps.begin(), voxelmaps.end());
  for (int i = 0; i < frames.size(); i++) {
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> deltas(frames.size());
    for (int j = 0; j < frames.size(); j++) {
      deltas[i] = Eigen::Isometry3d((poses_gt.at<gtsam::Pose3>(i).inverse() * poses_gt.at<gtsam::Pose3>(i)).matrix());
    }
    EXPECT_GT(gtsam_ext::overlap(voxelmaps_, frames[i], deltas), 0.99);
  }

  for (int i = 0; i < frames.size(); i++) {
    auto voxels = std::dynamic_pointer_cast<gtsam_ext::GaussianVoxelMapCPU>(voxelmaps[i]);
    ASSERT_TRUE(voxels);

    voxels->save_compact("/tmp/voxelmap.bin");

    auto voxels_ = gtsam_ext::GaussianVoxelMapCPU::load("/tmp/voxelmap.bin");
    ASSERT_TRUE(voxels_);

    EXPECT_DOUBLE_EQ(voxels->voxel_resolution(), voxels_->voxel_resolution());
    EXPECT_EQ(voxels->voxels.size(), voxels_->voxels.size());

    for (const auto& voxel : voxels->voxels) {
      const auto found = voxels_->voxels.find(voxel.first);
      ASSERT_TRUE(found != voxels_->voxels.end());

      EXPECT_EQ(voxel.second->num_points, found->second->num_points);
      EXPECT_LT((voxel.second->mean - found->second->mean).norm(), 1e-3);
      EXPECT_LT((voxel.second->cov - found->second->cov).norm(), 1e-3);
    }
  }

  // Test for merge_frames
  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses_(poses_gt.size());
  for (int i = 0; i < poses_gt.size(); i++) {
    poses_[i] = Eigen::Isometry3d(poses_gt.at<gtsam::Pose3>(i).matrix());
  }
  std::vector<gtsam_ext::PointCloud::ConstPtr> frames_(frames.begin(), frames.end());
  auto merged = gtsam_ext::merge_frames(poses_, frames_, 0.2);
  validate_frame(merged);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}