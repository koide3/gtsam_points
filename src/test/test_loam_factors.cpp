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

#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/factors/integrated_loam_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/util/parallelism.hpp>

struct LOAMTestBase : public testing::Test {
  virtual void SetUp() {
    const std::string data_path = "./data/newer_01";

    std::ifstream ifs(data_path + "/graph.txt");
    EXPECT_EQ(ifs.is_open(), true) << "Failed to open " << data_path << "/graph.txt";

    std::mt19937 mt(0);
    for (int i = 0; i < 5; i++) {
      // Read poses
      std::string token;
      Eigen::Vector3d trans;
      Eigen::Quaterniond quat;
      ifs >> token >> trans.x() >> trans.y() >> trans.z() >> quat.x() >> quat.y() >> quat.z() >> quat.w();

      Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
      pose.translation() = trans;
      pose.linear() = quat.toRotationMatrix();
      poses.insert(i, gtsam::Pose3::Identity());
      poses_gt.insert(i, gtsam::Pose3(pose.matrix()));

      // Load points
      const std::string edge_path = (boost::format("%s/edges_%06d.bin") % data_path % (i * 10)).str();
      const std::string plane_path = (boost::format("%s/planes_%06d.bin") % data_path % (i * 10)).str();

      auto edge_points = gtsam_points::read_points(edge_path);
      auto plane_points = gtsam_points::read_points(plane_path);

      EXPECT_NE(edge_points.size(), true) << "Faile to read edge points";
      EXPECT_NE(plane_points.size(), true) << "Faile to read plane points";

      edge_frames.emplace_back(std::make_shared<gtsam_points::PointCloudCPU>(edge_points));
      plane_frames.emplace_back(std::make_shared<gtsam_points::PointCloudCPU>(plane_points));
      plane_frames.back() = gtsam_points::randomgrid_sampling(plane_frames.back(), 1.0, 5000.0 / plane_frames.back()->size(), mt);
    }
  }

  std::vector<gtsam_points::PointCloud::Ptr> edge_frames;
  std::vector<gtsam_points::PointCloud::Ptr> plane_frames;
  gtsam::Values poses;
  gtsam::Values poses_gt;
};

TEST_F(LOAMTestBase, LoadCheck) {
  EXPECT_EQ(edge_frames.size(), 5) << "Failed to load edge points";
  EXPECT_EQ(plane_frames.size(), 5) << "Failed to load plane points";
  EXPECT_EQ(poses.size(), 5) << "Failed to load GT poses";
  EXPECT_EQ(poses_gt.size(), 5) << "Failed to load GT poses";
}

class LOAMFactorTest : public LOAMTestBase, public testing::WithParamInterface<std::tuple<std::string, std::string>> {
public:
  gtsam::NonlinearFactor::shared_ptr create_factor(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const gtsam_points::PointCloud::ConstPtr& target_edges,
    const gtsam_points::PointCloud::ConstPtr& target_planes,
    const gtsam_points::PointCloud::ConstPtr& source_edges,
    const gtsam_points::PointCloud::ConstPtr& source_planes) {
    const auto param = GetParam();
    const std::string method = std::get<0>(param);
    const std::string parallelism = std::get<1>(param);
    const int num_threads = parallelism == "NONE" ? 1 : 2;

    gtsam::NonlinearFactor::shared_ptr factor;
    if (method == "LOAM") {
      auto f =
        gtsam::make_shared<gtsam_points::IntegratedLOAMFactor>(target_key, source_key, target_edges, target_planes, source_edges, source_planes);
      f->set_num_threads(num_threads);
      factor = f;
    } else if (method == "EDGE") {
      auto f = gtsam::make_shared<gtsam_points::IntegratedPointToEdgeFactor>(target_key, source_key, target_edges, source_edges);
      f->set_num_threads(num_threads);
      factor = f;
    } else if (method == "PLANE") {
      auto f = gtsam::make_shared<gtsam_points::IntegratedPointToPlaneFactor>(target_key, source_key, target_planes, source_planes);
      f->set_num_threads(num_threads);
      factor = f;
    }

    return factor;
  }

  void test_graph(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& values, const std::string& note = "") {
    gtsam_points::LevenbergMarquardtExtParams lm_params;
    lm_params.setRelativeErrorTol(1e-4);
    gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
    gtsam::Values result = optimizer.optimize();

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

INSTANTIATE_TEST_SUITE_P(
  gtsam_points,
  LOAMFactorTest,
  testing::Combine(testing::Values("EDGE", "PLANE", "LOAM"), testing::Values("NONE", "OMP", "TBB")),
  [](const auto& info) { return std::get<0>(info.param) + "_" + std::get<1>(info.param); });

TEST_P(LOAMFactorTest, AlignmentTest) {
  const auto param = GetParam();
  const std::string method = std::get<0>(param);
  const std::string parallelism = std::get<1>(param);

#ifndef GTSAM_POINTS_USE_TBB
  if (parallelism == "TBB") {
    std::cerr << "Skip test for TBB" << std::endl;
    return;
  }
#endif

  if (parallelism == "TBB") {
    gtsam_points::set_tbb_as_default();
  } else {
    gtsam_points::set_omp_as_default();
  }

  auto f = create_factor(0, 1, edge_frames[0], plane_frames[0], edge_frames[1], plane_frames[1]);
  if (f == nullptr) {
    std::cerr << "[          ] SKIP:" << std::get<0>(GetParam()) << std::endl;
    return;
  }

  for (int i = 0; i < 2; i++) {
    gtsam::Values values;
    gtsam::NonlinearFactorGraph graph;

    values.insert(i, poses.at(i));
    values.insert(i + 1, poses.at(i + 1));
    graph.add(create_factor(i, i + 1, edge_frames[i], plane_frames[i], edge_frames[i + 1], plane_frames[i + 1]));
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(i, poses.at<gtsam::Pose3>(i), gtsam::noiseModel::Isotropic::Precision(6, 1e6)));

    test_graph(graph, values, "FORWARD_TEST_" + std::to_string(i));

    graph.erase(graph.begin() + static_cast<int>(graph.size()) - 1);
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(i + 1, poses.at<gtsam::Pose3>(i + 1), gtsam::noiseModel::Isotropic::Precision(6, 1e6)));

    test_graph(graph, values, "BACKWARD_TEST_" + std::to_string(i));
  }

  gtsam::Values values;
  gtsam::NonlinearFactorGraph graph;
  for (int i = 0; i < 5; i++) {
    values.insert(i, poses.at(i));
  }

  for (int i = 1; i < 5; i++) {
    graph.add(create_factor(i - 1, i, edge_frames[i - 1], plane_frames[i - 1], edge_frames[i], plane_frames[i]));
  }

  graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, poses.at<gtsam::Pose3>(0), gtsam::noiseModel::Isotropic::Precision(6, 1e6)));
  test_graph(graph, values, "MULTI_FRAME");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}