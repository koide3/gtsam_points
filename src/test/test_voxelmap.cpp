#include <vector>
#include <iostream>
#include <unordered_set>
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

TEST_F(VoxelMapTestBase, VoxelMapCPU_Intensity) {
  for (int i = 0; i < frames.size(); i++) {
    // without intensities
    {
      const auto& frame = frames[i];
      EXPECT_FALSE(frames[i]->has_intensities());

      auto voxels = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(1.0);
      voxels->insert(*frame);

      auto intensities = voxels->voxel_intensities();
      EXPECT_EQ(intensities.size(), voxels->num_voxels());
      for (const auto intensity : intensities) {
        EXPECT_EQ(intensity, 0.0f) << "Intensity should be zero when inserting points without intensities";
      }
    }

    // with intensities
    {
      auto frame = gtsam_points::PointCloudCPU::clone(*frames[i]);
      std::vector<float> intensities(frame->size());
      for (int j = 0; j < frame->size(); j++) {
        intensities[j] = (j % 128) + 128.0f;  // [128.0, 255.0]
      }
      frame->add_intensities(intensities);
      EXPECT_TRUE(frame->has_intensities());

      auto voxels = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(1.0);
      voxels->insert(*frame);

      auto voxel_intensities = voxels->voxel_intensities();
      EXPECT_EQ(voxel_intensities.size(), voxels->num_voxels());
      for (const auto intensity : voxel_intensities) {
        // TODO : implement CPU intensity handling
        EXPECT_GT(intensity, 128.0f - 0.1f) << "Intensity should be greater than zero when inserting points with intensities";
        EXPECT_LT(intensity, 255.0f + 0.1f) << "Intensity should be less than or equal to 255.0";
      }
    }
  }
}

#ifdef GTSAM_POINTS_USE_CUDA

TEST_F(VoxelMapTestBase, VoxelMapGPU) {
  for (int i = 0; i < frames.size(); i++) {
    const double overlap_gpu = gtsam_points::overlap_gpu(voxelmaps_gpu[i], frames[i], Eigen::Isometry3d::Identity());
    const double overlap_auto = gtsam_points::overlap_auto(voxelmaps_gpu[i], frames[i], Eigen::Isometry3d::Identity());
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

TEST_F(VoxelMapTestBase, VoxelMapGPU_Intensity) {
  for (int i = 0; i < frames.size(); i++) {
    // without intensities
    {
      const auto& frame = frames[i];
      EXPECT_FALSE(frames[i]->has_intensities());

      auto voxels = std::make_shared<gtsam_points::GaussianVoxelMapGPU>(1.0);
      voxels->insert(*frame);

      auto intensities = gtsam_points::download_voxel_intensities(*voxels);
      EXPECT_EQ(intensities.size(), voxels->voxelmap_info.num_voxels);
      for (const auto intensity : intensities) {
        EXPECT_EQ(intensity, 0.0f) << "Intensity should be zero when inserting points without intensities";
      }
    }

    // with intensities
    {
      auto frame = gtsam_points::PointCloudGPU::clone(*frames[i]);
      std::vector<float> intensities(frame->size());
      for (int j = 0; j < frame->size(); j++) {
        intensities[j] = (j % 128) + 128.0f;  // [128.0, 255.0]
      }
      frame->add_intensities(intensities);
      EXPECT_TRUE(frame->has_intensities());

      auto voxels = std::make_shared<gtsam_points::GaussianVoxelMapGPU>(1.0);
      voxels->insert(*frame);

      auto voxel_intensities = gtsam_points::download_voxel_intensities(*voxels);
      EXPECT_EQ(voxel_intensities.size(), voxels->voxelmap_info.num_voxels);
      for (const auto intensity : voxel_intensities) {
        EXPECT_GT(intensity, 128.0f - 0.1f) << "Intensity should be greater than zero when inserting points with intensities";
        EXPECT_LT(intensity, 255.0f + 0.1f) << "Intensity should be less than or equal to 255.0";
      }
    }
  }
}

TEST_F(VoxelMapTestBase, VoxelMapGPU_IO) {
  std::vector<gtsam_points::GaussianVoxelMapGPU::ConstPtr> voxels_gpu(voxelmaps_gpu.size());
  std::transform(voxelmaps_gpu.begin(), voxelmaps_gpu.end(), voxels_gpu.begin(), [](const gtsam_points::GaussianVoxelMap::ConstPtr& v) {
    return std::dynamic_pointer_cast<const gtsam_points::GaussianVoxelMapGPU>(v);
  });
  ASSERT_EQ(std::all_of(voxels_gpu.begin(), voxels_gpu.end(), [](const auto& v) { return v != nullptr; }), true);

  std::vector<gtsam_points::GaussianVoxelMapGPU::ConstPtr> loaded_voxels_from_cpu(voxelmaps.size());
  std::vector<gtsam_points::GaussianVoxelMapGPU::ConstPtr> loaded_voxels_from_gpu(voxelmaps.size());
  for (int i = 0; i < voxelmaps.size(); i++) {
    auto v1 = std::dynamic_pointer_cast<gtsam_points::GaussianVoxelMapCPU>(voxelmaps[i]);
    auto v2 = std::dynamic_pointer_cast<gtsam_points::GaussianVoxelMapGPU>(voxelmaps_gpu[i]);

    voxelmaps[i]->save_compact("/tmp/voxelmap_cpu.bin");
    loaded_voxels_from_cpu[i] = gtsam_points::GaussianVoxelMapGPU::load("/tmp/voxelmap_cpu.bin");
    ASSERT_TRUE(loaded_voxels_from_cpu[i] != nullptr);

    voxelmaps_gpu[i]->save_compact("/tmp/voxelmap_gpu.bin");
    loaded_voxels_from_gpu[i] = gtsam_points::GaussianVoxelMapGPU::load("/tmp/voxelmap_gpu.bin");
    ASSERT_TRUE(loaded_voxels_from_gpu[i] != nullptr);

    auto loaded = gtsam_points::GaussianVoxelMapGPU::load("/tmp/voxelmap_gpu.bin");
    ASSERT_TRUE(loaded != nullptr);
  }

  for (int i = 0; i < voxelmaps.size(); i++) {
    const auto voxels_cpu = std::dynamic_pointer_cast<gtsam_points::GaussianVoxelMapCPU>(voxelmaps[i]);
    const auto voxels_gpu = std::dynamic_pointer_cast<gtsam_points::GaussianVoxelMapGPU>(voxelmaps_gpu[i]);

    EXPECT_DOUBLE_EQ(voxels_cpu->voxel_resolution(), loaded_voxels_from_cpu[i]->voxel_resolution());
    EXPECT_EQ(voxels_cpu->num_voxels(), loaded_voxels_from_cpu[i]->voxelmap_info.num_voxels);

    EXPECT_DOUBLE_EQ(voxels_gpu->voxel_resolution(), loaded_voxels_from_gpu[i]->voxel_resolution());
    EXPECT_EQ(voxels_gpu->voxelmap_info.num_voxels, loaded_voxels_from_gpu[i]->voxelmap_info.num_voxels);

    // Verify means/covs for CPU-write GPU-read
    const auto means_cpu = voxels_cpu->voxel_points();
    const auto covs_cpu = voxels_cpu->voxel_covs();
    const auto loaded_means_cpu = gtsam_points::download_voxel_means(*loaded_voxels_from_cpu[i]);
    const auto loaded_covs_cpu = gtsam_points::download_voxel_covs(*loaded_voxels_from_cpu[i]);

    ASSERT_EQ(means_cpu.size(), loaded_means_cpu.size());
    ASSERT_EQ(covs_cpu.size(), loaded_covs_cpu.size());
    for (int j = 0; j < means_cpu.size(); j++) {
      EXPECT_LT((means_cpu[j].cast<float>().head<3>() - loaded_means_cpu[j]).cwiseAbs().maxCoeff(), 1e-3);
      EXPECT_LT((covs_cpu[j].cast<float>().topLeftCorner<3, 3>() - loaded_covs_cpu[j]).cwiseAbs().maxCoeff(), 1e-3);
    }

    // Verify buckets for CPU-write GPU-read
    const auto loaded_buckets_from_cpu = gtsam_points::download_buckets(*loaded_voxels_from_cpu[i]);
    std::unordered_map<Eigen::Vector3i, int, gtsam_points::Vector3iHash> map_from_cpu;
    for (const auto& b : loaded_buckets_from_cpu) {
      if (b.second < 0) {
        continue;
      }
      EXPECT_FALSE(map_from_cpu.count(b.first) > 0) << "Duplicate bucket found for coord " << b.first.transpose();
      EXPECT_GE(b.second, 0) << "Invalid bucket index found for coord " << b.first.transpose();
      EXPECT_LT(b.second, loaded_means_cpu.size()) << "Invalid bucket index found for coord " << b.first.transpose();

      map_from_cpu[b.first] = b.second;
    }

    for (const auto& mean : voxels_cpu->voxel_points()) {
      const Eigen::Vector3i coord = voxels_cpu->voxel_coord(mean);
      const auto found = map_from_cpu.find(coord);
      ASSERT_NE(found, map_from_cpu.end()) << "Failed to find bucket for coord " << coord.transpose();

      const auto& voxel_index = found->second;
      EXPECT_LE((mean.cast<float>().head<3>() - loaded_means_cpu[voxel_index]).cwiseAbs().maxCoeff(), 1e-3)
        << "Mean mismatch for coord " << coord.transpose();
    }

    // Verify means/covs for GPU-write GPU-read
    const auto means_gpu = gtsam_points::download_voxel_means(*voxels_gpu);
    const auto covs_gpu = gtsam_points::download_voxel_covs(*voxels_gpu);
    const auto loaded_means_gpu = gtsam_points::download_voxel_means(*loaded_voxels_from_gpu[i]);
    const auto loaded_covs_gpu = gtsam_points::download_voxel_covs(*loaded_voxels_from_gpu[i]);

    ASSERT_EQ(means_gpu.size(), loaded_means_gpu.size());
    ASSERT_EQ(covs_gpu.size(), loaded_covs_gpu.size());
    for (int j = 0; j < means_gpu.size(); j++) {
      EXPECT_LT((means_gpu[j] - loaded_means_gpu[j]).cwiseAbs().maxCoeff(), 1e-3);
      EXPECT_LT((covs_gpu[j] - loaded_covs_gpu[j]).cwiseAbs().maxCoeff(), 1e-3);
    }

    // Verify buckets for GPU-write GPU-read
    const auto loaded_buckets_from_gpu = gtsam_points::download_buckets(*loaded_voxels_from_gpu[i]);
    std::unordered_map<Eigen::Vector3i, int, gtsam_points::Vector3iHash> map_from_gpu;
    for (const auto& b : loaded_buckets_from_gpu) {
      if (b.second < 0) {
        continue;
      }
      EXPECT_FALSE(map_from_gpu.count(b.first) > 0) << "Duplicate bucket found for coord " << b.first.transpose();
      EXPECT_GE(b.second, 0) << "Invalid bucket index found for coord " << b.first.transpose();
      EXPECT_LT(b.second, loaded_means_gpu.size()) << "Invalid bucket index found for coord " << b.first.transpose();

      map_from_gpu[b.first] = b.second;
    }

    for (const auto& mean : means_gpu) {
      const Eigen::Vector3i coord = (mean.array() / voxels_gpu->voxel_resolution()).floor().cast<int>();
      const auto found = map_from_gpu.find(coord);
      ASSERT_NE(found, map_from_gpu.end()) << "Failed to find bucket for coord " << coord.transpose();

      const auto& voxel_index = found->second;
      EXPECT_LE((mean - loaded_means_gpu[voxel_index]).cwiseAbs().maxCoeff(), 1e-3) << "Mean mismatch for coord " << coord.transpose();
    }
  }

  // Verify overlap values with identity transform
  for (int i = 0; i < frames.size(); i++) {
    const double overlap_from_cpu = gtsam_points::overlap_gpu(loaded_voxels_from_cpu[i], frames[i], Eigen::Isometry3d::Identity());
    const double overlap_from_gpu = gtsam_points::overlap_gpu(loaded_voxels_from_gpu[i], frames[i], Eigen::Isometry3d::Identity());
    EXPECT_GT(overlap_from_cpu, 0.99) << "Overlap from CPU: " << overlap_from_cpu;
    EXPECT_GT(overlap_from_gpu, 0.99) << "Overlap from GPU: " << overlap_from_gpu;
  }

  // Verify overlap values with random transform
  for (int i = 0; i < frames.size(); i++) {
    Eigen::Isometry3d delta = Eigen::Isometry3d::Identity();
    delta.linear() = Eigen::AngleAxisd(Eigen::Vector2d::Random()[0] * 0.2, Eigen::Vector3d::Random().normalized()).toRotationMatrix();
    delta.translation() = Eigen::Vector3d::Random();

    const double overlap_cpu_orig = gtsam_points::overlap(voxelmaps[i], frames[i], delta);
    const double overlap_cpu_loaded = gtsam_points::overlap_gpu(loaded_voxels_from_cpu[i], frames[i], delta);
    EXPECT_NEAR(overlap_cpu_orig, overlap_cpu_loaded, 1e-3) << "Overlap mismatch CPU: " << overlap_cpu_orig << " GPU: " << overlap_cpu_loaded;

    const double overlap_gpu_orig = gtsam_points::overlap_gpu(voxelmaps_gpu[i], frames[i], delta);
    const double overlap_gpu_loaded = gtsam_points::overlap_gpu(loaded_voxels_from_gpu[i], frames[i], delta);
    EXPECT_NEAR(overlap_gpu_orig, overlap_gpu_loaded, 1e-3) << "Overlap mismatch GPU: " << overlap_gpu_orig << " GPU: " << overlap_gpu_loaded;
  }
}

#endif

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}