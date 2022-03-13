#include <random>
#include <string>
#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/format.hpp>

#include <gtest/gtest.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam_ext/util/read_points.hpp>
#include <gtsam_ext/util/covariance_estimation.hpp>
#include <gtsam_ext/types/voxelized_frame_cpu.hpp>
#include <gtsam_ext/types/voxelized_frame_gpu.hpp>
#include <gtsam_ext/factors/integrated_icp_factor.hpp>
#include <gtsam_ext/factors/integrated_gicp_factor.hpp>
#include <gtsam_ext/factors/integrated_vgicp_factor.hpp>
#include <gtsam_ext/factors/integrated_vgicp_factor_gpu.hpp>
#include <gtsam_ext/cuda/stream_temp_buffer_roundrobin.hpp>

#include <gtsam_ext/optimizers/isam2_ext.hpp>
#include <gtsam_ext/optimizers/levenberg_marquardt_ext.hpp>

struct MatchingCostFactorsTestBase : public testing::Test {
  virtual void SetUp() {
    std::string dump_path = "./data/kitti_07_dump";
    std::ifstream ifs(dump_path + "/graph.txt");
    EXPECT_EQ(ifs.is_open(), true) << "Failed to open " << dump_path;

    // It seems generated random numbers change depending on the compiler
    // Should we pregenerated randoms for reproductivity?
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

      std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points(points_f.size());
      std::transform(points_f.begin(), points_f.end(), points.begin(), [](const Eigen::Vector3f& p) {
        return Eigen::Vector4d(p[0], p[1], p[2], 1.0);
      });
      std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> covs = gtsam_ext::estimate_covariances(points);

#ifndef BUILD_GTSAM_EXT_GPU
      frames.push_back(gtsam_ext::VoxelizedFrame::Ptr(new gtsam_ext::VoxelizedFrameCPU(1.0, points, covs)));
#else
      frames.push_back(gtsam_ext::VoxelizedFrame::Ptr(new gtsam_ext::VoxelizedFrameGPU(1.0, points, covs)));
#endif
    }

#ifdef BUILD_GTSAM_EXT_GPU
    stream_buffer_roundrobin.reset(new gtsam_ext::StreamTempBufferRoundRobin(32));
#endif
  }

  std::vector<gtsam_ext::VoxelizedFrame::Ptr> frames;
  gtsam::Values poses;
  gtsam::Values poses_gt;

#ifdef BUILD_GTSAM_EXT_GPU
  std::unique_ptr<gtsam_ext::StreamTempBufferRoundRobin> stream_buffer_roundrobin;
#endif
};

TEST_F(MatchingCostFactorsTestBase, LoadCheck) {
  ASSERT_EQ(poses.size(), 5) << "Failed to load submap poses";
  ASSERT_EQ(poses_gt.size(), 5) << "Failed to load submap poses";
}

class MatchingCostFactorTest : public MatchingCostFactorsTestBase, public testing::WithParamInterface<std::string> {
public:
  gtsam::NonlinearFactor::shared_ptr create_factor(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const gtsam_ext::VoxelizedFrame::ConstPtr& target,
    const gtsam_ext::VoxelizedFrame::ConstPtr& source) {
    std::string method = GetParam();

    gtsam::NonlinearFactor::shared_ptr factor;
    if (method == "ICP") {
      factor.reset(new gtsam_ext::IntegratedICPFactor<>(target_key, source_key, target, source));
    } else if (method == "GICP") {
      factor.reset(new gtsam_ext::IntegratedGICPFactor<>(target_key, source_key, target, source));
    } else if (method == "VGICP") {
      factor.reset(new gtsam_ext::IntegratedVGICPFactor<>(target_key, source_key, target, source));
    } else if (method == "VGICP_CUDA") {
#ifdef BUILD_GTSAM_EXT_GPU
      auto stream_buffer = stream_buffer_roundrobin->get_stream_buffer();
      const auto& stream = stream_buffer.first;
      const auto& buffer = stream_buffer.second;
      factor.reset(new gtsam_ext::IntegratedVGICPFactorGPU(target_key, source_key, target, source, stream, buffer));
#endif
    }

    return factor;
  }

  gtsam::NonlinearFactor::shared_ptr create_factor(
    const gtsam::Pose3& fixed_target_pose,
    gtsam::Key source_key,
    const gtsam_ext::VoxelizedFrame::ConstPtr& target,
    const gtsam_ext::VoxelizedFrame::ConstPtr& source) {
    std::string method = GetParam();

    gtsam::NonlinearFactor::shared_ptr factor;
    if (method == "ICP") {
      factor.reset(new gtsam_ext::IntegratedICPFactor<>(fixed_target_pose, source_key, target, source));
    } else if (method == "GICP") {
      factor.reset(new gtsam_ext::IntegratedGICPFactor<>(fixed_target_pose, source_key, target, source));
    } else if (method == "VGICP") {
      factor.reset(new gtsam_ext::IntegratedVGICPFactor<>(fixed_target_pose, source_key, target, source));
    } else if (method == "VGICP_CUDA") {
#ifdef BUILD_GTSAM_EXT_GPU
      auto stream_buffer = stream_buffer_roundrobin->get_stream_buffer();
      const auto& stream = stream_buffer.first;
      const auto& buffer = stream_buffer.second;
      factor.reset(new gtsam_ext::IntegratedVGICPFactorGPU(fixed_target_pose, source_key, target, source, stream, buffer));
#endif
    }

    return factor;
  }

  void test_graph(
    const gtsam::NonlinearFactorGraph& graph,
    const gtsam::Values& values,
    const gtsam::Values& additional_values,
    const std::string& note = "") {
    gtsam_ext::LevenbergMarquardtExtParams lm_params;
    gtsam_ext::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
    gtsam::Values result = optimizer.optimize();
    result.insert(additional_values);

    bool is_first = true;
    gtsam::Pose3 delta;

    for (const auto& value : result) {
      const gtsam::Pose3 pose_gt = poses_gt.at<gtsam::Pose3>(value.key);
      const gtsam::Pose3 pose = value.value.cast<gtsam::Pose3>();

      if (is_first) {
        is_first = false;
        delta = pose_gt * pose.inverse();
        continue;
      }

      const gtsam::Pose3 pose_error = pose_gt.inverse() * (delta * pose);
      const gtsam::Vector6 error = gtsam::Pose3::Logmap(pose_error);
      double error_r = error.head<3>().norm();
      double error_t = error.tail<3>().norm();

      EXPECT_LT(error_r, 0.015) << "Too large rotation error " << note;
      EXPECT_LT(error_t, 0.15) << "Too large translation error " << note;
    }
  }
};

INSTANTIATE_TEST_SUITE_P(gtsam_ext, MatchingCostFactorTest, testing::Values("ICP", "GICP", "VGICP", "VGICP_CUDA"), [](const auto& info) {
  return info.param;
});

TEST_P(MatchingCostFactorTest, AlignmentTest) {
  auto f = create_factor(0, 1, frames[0], frames[1]);
  if (f == nullptr) {
    std::cerr << "[          ] SKIP:" << GetParam() << std::endl;
    return;
  }

  for (int i = 0; i < 4; i++) {
    gtsam::Values values;
    gtsam::NonlinearFactorGraph graph;

    values.insert(i, poses.at(i));
    values.insert(i + 1, poses.at(i + 1));
    graph.add(create_factor(i, i + 1, frames[i], frames[i + 1]));
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(i, poses.at<gtsam::Pose3>(i), gtsam::noiseModel::Isotropic::Precision(6, 1e6)));

    test_graph(graph, values, gtsam::Values(), "FORWARD_TEST_" + std::to_string(i));

    graph.erase(graph.begin() + static_cast<int>(graph.size()) - 1);
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(i + 1, poses.at<gtsam::Pose3>(i + 1), gtsam::noiseModel::Isotropic::Precision(6, 1e6)));

    test_graph(graph, values, gtsam::Values(), "BACKWARD_TEST_" + std::to_string(i));
  }

  for (int i = 0; i < 4; i++) {
    gtsam::Values values;
    gtsam::Values fixed_values;
    gtsam::NonlinearFactorGraph graph;

    fixed_values.insert(i, poses.at<gtsam::Pose3>(i));
    values.insert(i + 1, poses.at<gtsam::Pose3>(i + 1));
    graph.add(create_factor(poses.at<gtsam::Pose3>(i), i + 1, frames[i], frames[i + 1]));

    test_graph(graph, values, fixed_values, "UNARY_TEST_" + std::to_string(i));
  }

  gtsam::Values values;
  gtsam::NonlinearFactorGraph graph;
  for (int i = 0; i < 5; i++) {
    values.insert(i, poses.at(i));
    for (int j = i + 1; j < 5; j++) {
      graph.add(create_factor(i, j, frames[i], frames[j]));
    }
  }
  graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, poses.at<gtsam::Pose3>(0), gtsam::noiseModel::Isotropic::Precision(6, 1e6)));
  test_graph(graph, values, gtsam::Values(), "MULTI_FRAME");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}