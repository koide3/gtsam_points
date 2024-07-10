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
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/factors/balm_feature.hpp>
#include <gtsam_points/factors/bundle_adjustment_factor_evm.hpp>
#include <gtsam_points/factors/bundle_adjustment_factor_lsq.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/util/numerical.hpp>

TEST(BATest, DerivativeTest) {
  std::mt19937 mt(4096 - 1);
  std::normal_distribution<> ndist;

  for (int num_points = 6; num_points <= 32; num_points *= 2) {
    std::vector<Eigen::Vector3d> points;
    for (int i = 0; i < num_points; i++) {
      points.push_back(Eigen::Vector3d(ndist(mt) * 0.1, ndist(mt) * 0.4, ndist(mt)));
    }

    gtsam_points::BALMFeature feature(points);

    Eigen::VectorXd Ja0(3 * num_points);
    Eigen::VectorXd Ja1(3 * num_points);
    Eigen::VectorXd Ja2(3 * num_points);
    Eigen::MatrixXd Ha0(3 * num_points, 3 * num_points);
    Eigen::MatrixXd Ha1(3 * num_points, 3 * num_points);
    Eigen::MatrixXd Ha2(3 * num_points, 3 * num_points);
    for (int i = 0; i < num_points; i++) {
      Ja0.block<3, 1>(i * 3, 0) = feature.Ji<0>(points[i]);
      Ja1.block<3, 1>(i * 3, 0) = feature.Ji<1>(points[i]);
      Ja2.block<3, 1>(i * 3, 0) = feature.Ji<2>(points[i]);
      for (int j = 0; j < num_points; j++) {
        Ha0.block<3, 3>(i * 3, j * 3) = feature.Hij<0>(points[i], points[j], i == j);
        Ha1.block<3, 3>(i * 3, j * 3) = feature.Hij<1>(points[i], points[j], i == j);
        Ha2.block<3, 3>(i * 3, j * 3) = feature.Hij<2>(points[i], points[j], i == j);
      }
    }

    const auto calc_eigenvalue = [](int k, const Eigen::VectorXd& x) {
      std::vector<Eigen::Vector3d> points(x.size() / 3);
      for (int i = 0; i < points.size(); i++) {
        points[i] = x.block<3, 1>(i * 3, 0);
      }
      return gtsam_points::BALMFeature(points).eigenvalues[k];
    };

    Eigen::VectorXd x0 = Eigen::Map<Eigen::VectorXd>(points[0].data(), 3 * num_points);
    Eigen::VectorXd Jn0 = gtsam_points::numerical_jacobian([&](const Eigen::VectorXd& x) { return calc_eigenvalue(0, x); }, x0);
    Eigen::MatrixXd Hn0 = gtsam_points::numerical_hessian([&](const Eigen::VectorXd& x) { return calc_eigenvalue(0, x); }, x0);
    Eigen::VectorXd Jn1 = gtsam_points::numerical_jacobian([&](const Eigen::VectorXd& x) { return calc_eigenvalue(1, x); }, x0);
    Eigen::MatrixXd Hn1 = gtsam_points::numerical_hessian([&](const Eigen::VectorXd& x) { return calc_eigenvalue(1, x); }, x0);
    Eigen::VectorXd Jn2 = gtsam_points::numerical_jacobian([&](const Eigen::VectorXd& x) { return calc_eigenvalue(2, x); }, x0);
    Eigen::MatrixXd Hn2 = gtsam_points::numerical_hessian([&](const Eigen::VectorXd& x) { return calc_eigenvalue(2, x); }, x0);

    const double err_J0 = (Jn0 - Ja0).array().abs().maxCoeff();
    const double err_H0 = (Hn0 - Ha0).array().abs().maxCoeff();
    const double err_J1 = (Jn1 - Ja1).array().abs().maxCoeff();
    const double err_H1 = (Hn1 - Ha1).array().abs().maxCoeff();
    const double err_J2 = (Jn2 - Ja2).array().abs().maxCoeff();
    const double err_H2 = (Hn2 - Ha2).array().abs().maxCoeff();

    EXPECT_LT(err_J0, 1e-5) << "Too large Jacobian error for lambda_0 " << num_points << "[pts]";
    EXPECT_LT(err_H0, 1e-3) << "Too large Hessian error for lambda_0 " << num_points << "[pts]";
    EXPECT_LT(err_J1, 1e-5) << "Too large Jacobian error for lambda_1 " << num_points << "[pts]";
    EXPECT_LT(err_H1, 1e-3) << "Too large Hessian error for lambda_1 " << num_points << "[pts]";
    EXPECT_LT(err_J2, 1e-5) << "Too large Jacobian error for lambda_2 " << num_points << "[pts]";
    EXPECT_LT(err_H2, 2e-3) << "Too large Hessian error for lambda_2 " << num_points << "[pts]";
  }
}

struct BATestBase : public testing::Test {
  virtual void SetUp() {
    const std::string data_path = "./data/newer_01";

    std::ifstream ifs(data_path + "/graph.txt");
    EXPECT_EQ(ifs.is_open(), true) << "Failed to open " << data_path << "/graph.txt";

    gtsam::Values pose_noises;
    pose_noises.insert(0, gtsam::Pose3::Identity());
    pose_noises.insert(1, gtsam::Pose3::Expmap((gtsam::Vector6() << 0.02, 0.02, 0.02, 0.2, 0.2, 0.2).finished()));
    pose_noises.insert(2, gtsam::Pose3::Expmap((gtsam::Vector6() << -0.02, 0.02, 0.02, -0.2, 0.2, 0.2).finished()));
    pose_noises.insert(3, gtsam::Pose3::Expmap((gtsam::Vector6() << 0.02, -0.02, 0.02, 0.2, -0.2, 0.2).finished()));
    pose_noises.insert(4, gtsam::Pose3::Expmap((gtsam::Vector6() << 0.02, 0.02, -0.02, 0.2, 0.2, -0.2).finished()));

    for (int i = 0; i < 5; i++) {
      // Read poses
      std::string token;
      Eigen::Vector3d trans;
      Eigen::Quaterniond quat;
      ifs >> token >> trans.x() >> trans.y() >> trans.z() >> quat.x() >> quat.y() >> quat.z() >> quat.w();

      Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
      pose.translation() = trans;
      pose.linear() = quat.toRotationMatrix();
      poses.insert(i, gtsam::Pose3(pose.matrix()) * pose_noises.at<gtsam::Pose3>(i));
      poses_gt.insert(i, gtsam::Pose3(pose.matrix()));

      // Load points
      const std::string edge_path = (boost::format("%s/edges_%06d.bin") % data_path % (i * 10)).str();
      const std::string plane_path = (boost::format("%s/planes_%06d.bin") % data_path % (i * 10)).str();

      auto edge_points = gtsam_points::read_points(edge_path);
      auto plane_points = gtsam_points::read_points(plane_path);

      EXPECT_NE(edge_points.size(), true) << "Faile to read edge points";
      EXPECT_NE(plane_points.size(), true) << "Faile to read plane points";

      edge_frames.push_back(gtsam_points::PointCloud::Ptr(new gtsam_points::PointCloudCPU(edge_points)));
      plane_frames.push_back(gtsam_points::PointCloud::Ptr(new gtsam_points::PointCloudCPU(plane_points)));
    }
  }

  std::vector<gtsam_points::PointCloud::Ptr> edge_frames;
  std::vector<gtsam_points::PointCloud::Ptr> plane_frames;
  gtsam::Values poses;
  gtsam::Values poses_gt;
};

TEST_F(BATestBase, LoadCheck) {
  ASSERT_EQ(edge_frames.size(), 5) << "Failed to load edge points";
  ASSERT_EQ(plane_frames.size(), 5) << "Failed to load plane points";
  ASSERT_EQ(poses.size(), 5) << "Failed to load GT poses";
  ASSERT_EQ(poses_gt.size(), 5) << "Failed to load GT poses";
}

class BAFactorTest : public BATestBase, public testing::WithParamInterface<std::string> {
public:
  void test_result(const gtsam::Values& result, const std::string& note = "") {
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
      EXPECT_LT(error_t, 0.20) << "Too large translation error " << note;
    }
  }
};

INSTANTIATE_TEST_SUITE_P(gtsam_points, BAFactorTest, testing::Values("EVM", "LSQ"), [](const auto& info) { return info.param; });

TEST_P(BAFactorTest, AlignmentTest) {
  gtsam::Values values = poses;
  gtsam::NonlinearFactorGraph graph;
  graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3::Identity(), gtsam::noiseModel::Isotropic::Precision(6, 1e3)));

  std::vector<Eigen::Vector3d> plane_centers;
  plane_centers.push_back(Eigen::Vector3d(3.16, 1.79, -1.30));
  plane_centers.push_back(Eigen::Vector3d(25.44, 8.26, 3.68));
  plane_centers.push_back(Eigen::Vector3d(25.41, 19.51, 4.36));
  plane_centers.push_back(Eigen::Vector3d(27.23, 27.48, 8.04));
  plane_centers.push_back(Eigen::Vector3d(5.39, 14.86, 3.23));
  plane_centers.push_back(Eigen::Vector3d(1.05, -13.85, -1.98));
  plane_centers.push_back(Eigen::Vector3d(-7.39, 3.93, -0.25));
  plane_centers.push_back(Eigen::Vector3d(11.15, 12.25, -0.45));
  plane_centers.push_back(Eigen::Vector3d(11.17, -22.04, 4.52));

  std::vector<Eigen::Vector3d> edge_centers;
  edge_centers.push_back(Eigen::Vector3d(12.11, -10.86, 1.07));
  edge_centers.push_back(Eigen::Vector3d(14.45, -7.17, 1.09));
  edge_centers.push_back(Eigen::Vector3d(22.30, 15.17, 2.38));
  edge_centers.push_back(Eigen::Vector3d(16.92, 19.16, 2.98));
  edge_centers.push_back(Eigen::Vector3d(-4.82, 1.31, -0.09));
  edge_centers.push_back(Eigen::Vector3d(-17.10, -7.10, 0.52));

  // Create plane factors
  gtsam::NonlinearFactorGraph plane_factors;
  for (const auto& center : plane_centers) {
    gtsam_points::BundleAdjustmentFactorBase::shared_ptr factor;

    if (GetParam() == "EVM") {
      factor.reset(new gtsam_points::PlaneEVMFactor());
    } else if (GetParam() == "LSQ") {
      factor.reset(new gtsam_points::LsqBundleAdjustmentFactor());
    }

    for (int i = 0; i < plane_frames.size(); i++) {
      for (int j = 0; j < plane_frames[i]->size(); j++) {
        const Eigen::Vector3d pt = plane_frames[i]->points[j].head<3>();
        const Eigen::Vector3d transed_pt = values.at<gtsam::Pose3>(i) * pt;
        if ((transed_pt - center).norm() < 1.0) {
          factor->add(pt, i);
        }
      }
    }

    plane_factors.add(factor);
  }

  // Create edge factors
  gtsam::NonlinearFactorGraph edge_factors;
  for (const auto& center : edge_centers) {
    gtsam_points::BundleAdjustmentFactorBase::shared_ptr factor(new gtsam_points::EdgeEVMFactor());
    for (int i = 0; i < edge_frames.size(); i++) {
      for (int j = 0; j < edge_frames[i]->size(); j++) {
        const Eigen::Vector3d pt = edge_frames[i]->points[j].head<3>();
        const Eigen::Vector3d transed_pt = values.at<gtsam::Pose3>(i) * pt;
        if ((transed_pt - center).norm() < 1.0) {
          factor->add(pt, i);
        }
      }
    }

    edge_factors.add(factor);
  }

  graph.add(plane_factors);
  graph.add(edge_factors);

  gtsam::LevenbergMarquardtParams lm_params;
  gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, lm_params);
  values = optimizer.optimize();

  test_result(values);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}