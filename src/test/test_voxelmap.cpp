#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <boost/format.hpp>

#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam_points/config.hpp>
#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/util/easy_profiler.hpp>
#include <gtsam_points/features/covariance_estimation.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/types/point_cloud_gpu.hpp>
#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_points/types/gaussian_voxelmap_gpu.hpp>

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
      auto points_f = gtsam_points::read_points(points_path);
      EXPECT_NE(points_f.empty(), true) << "Failed to read points";

      auto frame = std::make_shared<gtsam_points::PointCloudCPU>(points_f);
      frame = gtsam_points::randomgrid_sampling(frame, 1.0, 10000.0 / frame->size(), mt);
      frame->add_covs(gtsam_points::estimate_covariances(frame->points, frame->size()));
      frames.push_back(frame);

#ifdef GTSAM_POINTS_USE_CUDA
      frames.back() = gtsam_points::PointCloudGPU::clone(*frames.back());
#endif

      auto voxelmap = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(1.0);
      voxelmap->insert(*frames.back());
      voxelmaps.push_back(voxelmap);

#ifdef GTSAM_POINTS_USE_CUDA
      auto voxelmap_gpu = std::make_shared<gtsam_points::GaussianVoxelMapGPU>(1.0);
      voxelmap_gpu->insert(*frames.back());
      voxelmaps_gpu.push_back(voxelmap_gpu);
#else
      voxelmaps_gpu.push_back(nullptr);
#endif
    }
  }

  std::vector<gtsam_points::PointCloud::Ptr> frames;
  std::vector<gtsam_points::GaussianVoxelMap::Ptr> voxelmaps;
  std::vector<gtsam_points::GaussianVoxelMap::Ptr> voxelmaps_gpu;
  gtsam::Values poses;
  gtsam::Values poses_gt;
};

TEST_F(VoxelMapTestBase, LoadCheck) {
  ASSERT_EQ(poses.size(), 5) << "Failed to load submap poses";
  ASSERT_EQ(poses_gt.size(), 5) << "Failed to load submap poses";
}

TEST_F(VoxelMapTestBase, VoxelMapCPU) {
  for (int i = 0; i < frames.size(); i++) {
    const double overlap = gtsam_points::overlap(voxelmaps[i], frames[i], Eigen::Isometry3d::Identity());
    const double overlap_auto = gtsam_points::overlap_auto(voxelmaps[i], frames[i], Eigen::Isometry3d::Identity());
    EXPECT_GT(overlap, 0.99);
    EXPECT_DOUBLE_EQ(overlap, overlap_auto);
  }

  std::vector<gtsam_points::GaussianVoxelMap::ConstPtr> voxelmaps_(voxelmaps.begin(), voxelmaps.end());
  for (int i = 0; i < frames.size(); i++) {
    std::vector<Eigen::Isometry3d> deltas(frames.size());
    for (int j = 0; j < frames.size(); j++) {
      deltas[i] = Eigen::Isometry3d((poses_gt.at<gtsam::Pose3>(i).inverse() * poses_gt.at<gtsam::Pose3>(i)).matrix());
    }
    const double overlap = gtsam_points::overlap(voxelmaps_, frames[i], deltas);
    const double overlap_auto = gtsam_points::overlap_auto(voxelmaps_, frames[i], deltas);
    EXPECT_GT(overlap, 0.99);
    EXPECT_DOUBLE_EQ(overlap, overlap_auto);
  }

  for (int i = 0; i < frames.size(); i++) {
    auto voxels = std::dynamic_pointer_cast<gtsam_points::GaussianVoxelMapCPU>(voxelmaps[i]);
    ASSERT_TRUE(voxels);

    voxels->save_compact("/tmp/voxelmap.bin");

    auto voxels_ = gtsam_points::GaussianVoxelMapCPU::load("/tmp/voxelmap.bin");
    ASSERT_TRUE(voxels_);

    EXPECT_DOUBLE_EQ(voxels->voxel_resolution(), voxels_->voxel_resolution());
    EXPECT_EQ(voxels->num_voxels(), voxels_->num_voxels());

    const auto points = voxels->voxel_points();
    const auto points_ = voxels_->voxel_points();
    EXPECT_EQ(points.size(), points_.size());
    for (int i = 0; i < points.size(); i++) {
      EXPECT_LT((points[i] - points_[i]).norm(), 1e-3);
    }

    const auto covs = voxels->voxel_covs();
    const auto covs_ = voxels_->voxel_covs();
    EXPECT_EQ(covs.size(), covs_.size());
    for (int i = 0; i < covs.size(); i++) {
      EXPECT_LT((covs[i] - covs_[i]).norm(), 1e-3);
    }

    for (const auto& pt : points) {
      const Eigen::Vector3i coord = voxels->voxel_coord(pt);
      const Eigen::Vector3i coord_ = voxels_->voxel_coord(pt);
      EXPECT_EQ(coord, coord_);

      const auto index = voxels->lookup_voxel_index(coord);
      const auto index_ = voxels_->lookup_voxel_index(coord);
      EXPECT_EQ(index, index_);

      const auto voxel = voxels->lookup_voxel(index);
      const auto voxel_ = voxels_->lookup_voxel(index);
      EXPECT_EQ(voxel.num_points, voxel_.num_points);
      EXPECT_LT((voxel.mean - voxel_.mean).norm(), 1e-3);
      EXPECT_LT((voxel.cov - voxel_.cov).norm(), 1e-3);
    }
  }

  // Test for merge_frames
  std::vector<Eigen::Isometry3d> poses_(poses_gt.size());
  for (int i = 0; i < poses_gt.size(); i++) {
    poses_[i] = Eigen::Isometry3d(poses_gt.at<gtsam::Pose3>(i).matrix());
  }
  std::vector<gtsam_points::PointCloud::ConstPtr> frames_(frames.begin(), frames.end());
  auto merged = gtsam_points::merge_frames(poses_, frames_, 0.2);
  validate_frame(merged);

  auto merged2 = gtsam_points::merge_frames_auto(poses_, frames_, 0.2);
  validate_frame(merged2);
}

#ifdef GTSAM_POINTS_USE_CUDA

TEST_F(VoxelMapTestBase, VoxelMapGPU) {
  for (int i = 0; i < frames.size(); i++) {
    const double overlap_gpu = gtsam_points::overlap_gpu(voxelmaps_gpu[i], frames[i], Eigen::Isometry3d::Identity());
    const double overlap_auto = gtsam_points::overlap_gpu(voxelmaps_gpu[i], frames[i], Eigen::Isometry3d::Identity());
    EXPECT_GE(overlap_gpu, 0.99);
    EXPECT_DOUBLE_EQ(overlap_gpu, overlap_auto);
  }

  std::vector<gtsam_points::GaussianVoxelMap::ConstPtr> voxelmaps_(voxelmaps_gpu.begin(), voxelmaps_gpu.end());
  for (int i = 0; i < frames.size(); i++) {
    std::vector<Eigen::Isometry3d> deltas(frames.size());
    for (int j = 0; j < frames.size(); j++) {
      deltas[i] = Eigen::Isometry3d((poses_gt.at<gtsam::Pose3>(i).inverse() * poses_gt.at<gtsam::Pose3>(i)).matrix());
    }
    const double overlap_gpu = gtsam_points::overlap_gpu(voxelmaps_, frames[i], deltas);
    const double overlap_auto = gtsam_points::overlap_auto(voxelmaps_, frames[i], deltas);
    EXPECT_GT(overlap_gpu, 0.99);
    EXPECT_DOUBLE_EQ(overlap_gpu, overlap_auto);
  }

  for (int i = 0; i < frames.size(); i++) {
    Eigen::Isometry3d delta = Eigen::Isometry3d::Identity();
    delta.linear() = Eigen::AngleAxisd(Eigen::Vector2d::Random()[0] * 0.2, Eigen::Vector3d::Random().normalized()).toRotationMatrix();
    delta.translation() = Eigen::Vector3d::Random();

    const double overlap_cpu = gtsam_points::overlap(voxelmaps[i], frames[i], delta);
    const double overlap_gpu = gtsam_points::overlap_gpu(voxelmaps_gpu[i], frames[i], delta);
    EXPECT_LT(std::abs(overlap_cpu - overlap_gpu), 0.01) << "Overlap mismatch CPU=" << overlap_cpu << " GPU=" << overlap_gpu;
  }

  for (int i = 0; i < frames.size(); i++) {
    auto voxelmap_gpu = std::dynamic_pointer_cast<gtsam_points::GaussianVoxelMapGPU>(voxelmaps_gpu[i]);
    const auto means = gtsam_points::download_voxel_means(*voxelmap_gpu);
    const auto covs = gtsam_points::download_voxel_covs(*voxelmap_gpu);
    EXPECT_EQ(means.size(), voxelmap_gpu->voxelmap_info.num_voxels);
    EXPECT_EQ(covs.size(), voxelmap_gpu->voxelmap_info.num_voxels);
    EXPECT_TRUE(std::all_of(means.begin(), means.end(), [](const Eigen::Vector3f& p) { return p.array().isFinite().all(); }));
    EXPECT_TRUE(std::all_of(covs.begin(), covs.end(), [](const Eigen::Matrix3f& c) { return c.array().isFinite().all(); }));
  }

  // Test for merge_frames
  std::vector<Eigen::Isometry3d> poses_(poses_gt.size());
  for (int i = 0; i < poses_gt.size(); i++) {
    poses_[i] = Eigen::Isometry3d(poses_gt.at<gtsam::Pose3>(i).matrix());
  }
  std::vector<gtsam_points::PointCloud::ConstPtr> frames_(frames.begin(), frames.end());
  auto merged = gtsam_points::merge_frames_gpu(poses_, frames_, 0.2);
  validate_frame_gpu(merged);
}

#endif

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}