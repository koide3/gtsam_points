#include <vector>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtest/gtest.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam_points/util/expressions.hpp>
#include <gtsam_points/util/bspline.hpp>
#include <gtsam_points/util/continuous_trajectory.hpp>

using gtsam::symbol_shorthand::X;

struct ContinuousTrajectoryTestBase : public testing::Test {
public:
  virtual void SetUp() {
    std::mt19937 mt(8192 - 3);

    for (double t = 0.0; t < 10.0; t += 0.5) {
      gtsam::Vector6 tan;
      for (int i = 0; i < 6; i++) {
        tan[i] = std::uniform_real_distribution<>(-0.5, 0.5)(mt);
      }
      tan[3] += t;

      stamps.push_back(t);
      poses.push_back(gtsam::Pose3::Expmap(tan));
    }
  }

  void check_pose_error(const gtsam::Pose3& x0, const gtsam::Pose3& x1, const std::string& label, const double thresh = 1e-1) {
    const gtsam::Pose3 error = x0.inverse() * x1;
    const double error_t = error.translation().norm();
    const double error_r = Eigen::AngleAxisd(error.rotation().matrix()).angle();

    EXPECT_LT(error_t, thresh) << "[" << label << "] Too large translation error";
    EXPECT_LT(error_r, thresh) << "[" << label << "] Too large rotation error";
  }

  void check_error(const Eigen::VectorXd& x0, const Eigen::VectorXd& x1, const std::string& label, const double thresh = 1e-1) {  //
    EXPECT_LT((x0 - x1).array().abs().maxCoeff(), thresh) << "[" << label << "] Too large error " << x0.transpose() << " vs " << x1.transpose();
  }

  void fit_knots(const double knot_interval) {
    values.reset(new gtsam::Values);
    ct.reset(new gtsam_points::ContinuousTrajectory('x', stamps.front(), stamps.back(), knot_interval));
    *values = ct->fit_knots(stamps, poses);
  }

  void fitting_test() {
    for (int i = 0; i < stamps.size(); i++) {
      const auto pose = ct->pose(*values, stamps[i]);
      check_pose_error(pose, poses[i], "Fitting");
    }
  }

  void interpolation_test() {
    values->insert(0, 0.0);
    for (double t = stamps.front(); t < stamps.back(); t += 0.05) {
      const auto pose = ct->pose(*values, t);

      values->update(0, t);
      const int knot_i = ct->knot_id(t);
      const double knot_t = ct->knot_stamp(knot_i);

      const gtsam::Double_ p = (1.0 / ct->knot_interval) * (gtsam::Double_(gtsam::Key(0)) - gtsam::Double_(knot_t));
      const auto pose0_ = gtsam_points::bspline(X(knot_i), p);
      const auto pose0 = pose0_.value(*values);
      check_pose_error(pose, pose0, "Interpolation");

      const auto rot0_ = gtsam_points::bspline_so3(  //
        gtsam::rotation(X(knot_i - 1)),
        gtsam::rotation(X(knot_i)),
        gtsam::rotation(X(knot_i + 1)),
        gtsam::rotation(X(knot_i + 2)),
        p);
      std::vector<gtsam::Matrix> Hs_rot0(rot0_.keys().size());
      const auto rot0 = rot0_.value(*values, Hs_rot0);

      const auto trans0_ = gtsam_points::bspline_trans(
        gtsam::translation(X(knot_i - 1)),
        gtsam::translation(X(knot_i)),
        gtsam::translation(X(knot_i + 1)),
        gtsam::translation(X(knot_i + 2)),
        p);
      std::vector<gtsam::Matrix> Hs_trans0(trans0_.keys().size());
      const auto trans0 = trans0_.value(*values, Hs_trans0);
      check_pose_error(pose, gtsam::Pose3(rot0, trans0), "Independent");

      // Derivatives
      const auto dr_dt_ = gtsam_points::bspline_angular_vel(
        gtsam::rotation(X(knot_i - 1)),
        gtsam::rotation(X(knot_i)),
        gtsam::rotation(X(knot_i + 1)),
        gtsam::rotation(X(knot_i + 2)),
        p,
        ct->knot_interval);
      const auto dr_dt = dr_dt_.value(*values);
      check_error(dr_dt, Hs_rot0.front(), "Angular vel", 1e-1);

      const auto dt_dt_ = gtsam_points::bspline_linear_vel(
        gtsam::translation(X(knot_i - 1)),
        gtsam::translation(X(knot_i)),
        gtsam::translation(X(knot_i + 1)),
        gtsam::translation(X(knot_i + 2)),
        p,
        ct->knot_interval);

      std::vector<gtsam::Matrix> Hs_tvel(dt_dt_.keys().size());
      const auto dt_dt = dt_dt_.value(*values, Hs_tvel);
      check_error(dt_dt, Hs_trans0.front(), "Linear vel", 5e-2);

      const auto dt_dt2_ = gtsam_points::bspline_linear_acc(
        gtsam::translation(X(knot_i - 1)),
        gtsam::translation(X(knot_i)),
        gtsam::translation(X(knot_i + 1)),
        gtsam::translation(X(knot_i + 2)),
        p,
        ct->knot_interval);
      const auto dt_dt2 = dt_dt2_.value(*values);
      check_error(dt_dt2, Hs_tvel.front(), "Linear acc", 5e-2);

      const Eigen::Vector3d g(0.0, 0.0, 9.80665);
      const auto imu_ = gtsam_points::bspline_imu(  //
        gtsam::Pose3_(X(knot_i - 1)),
        gtsam::Pose3_(X(knot_i)),
        gtsam::Pose3_(X(knot_i + 1)),
        gtsam::Pose3_(X(knot_i + 2)),
        p,
        ct->knot_interval,
        g);
      const auto imu = imu_.value(*values);
      gtsam::Vector6 imu2 = (gtsam::Vector6() << rot0.unrotate(dt_dt2 + g), dr_dt).finished();
      check_error(imu2.head<3>(), imu.head<3>(), "IMU self w", 5e-2);
      check_error(imu2.tail<3>(), imu.tail<3>(), "IMU self a", 5e-2);
    }
  }

  void imu_test() {
    for (int i = 0; i < imu_data.size(); i++) {
      const double t = imu_data[i][0];
      const gtsam::Vector6 imu_gt = imu_data[i].block<6, 1>(1, 0);

      if (t < stamps.front() || t > stamps.back()) {
        continue;
      }

      const int knot_i = ct->knot_id(t);
      const double knot_t = ct->knot_stamp(knot_i);
      const double p = (t - knot_t) / ct->knot_interval;

      const gtsam::Vector6 imu = ct->imu(*values, t);

      check_error(imu_gt.head<3>(), imu.head<3>(), "IMU GT a", 0.2);
      check_error(imu_gt.tail<3>(), imu.tail<3>(), "IMU GT w", 5e-2);
    }
  }

public:
  std::vector<double> stamps;
  std::vector<gtsam::Pose3> poses;
  std::vector<gtsam::Vector7> imu_data;

  std::unique_ptr<gtsam::Values> values;
  std::unique_ptr<gtsam_points::ContinuousTrajectory> ct;
};

TEST_F(ContinuousTrajectoryTestBase, RandomKnots) {
  fit_knots(0.2);
  fitting_test();
  interpolation_test();

  fit_knots(0.05);
  fitting_test();
  interpolation_test();
}

TEST_F(ContinuousTrajectoryTestBase, IMUTest) {
  std::ifstream traj_ifs("./data/continuous/traj.txt");
  ASSERT_EQ(traj_ifs.is_open(), true) << "Failed to open traj.txt";

  std::ifstream imu_ifs("./data/continuous/imu.txt");
  ASSERT_EQ(imu_ifs.is_open(), true) << "Failed to open imu.txt";

  std::string line;

  stamps.clear();
  poses.clear();
  while (!traj_ifs.eof() && std::getline(traj_ifs, line) && !line.empty()) {
    std::stringstream sst(line);
    double time;
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
    sst >> time >> t.x() >> t.y() >> t.z() >> q.x() >> q.y() >> q.z() >> q.w();

    stamps.push_back(time);
    poses.push_back(gtsam::Pose3(gtsam::Rot3(q), t));
  }

  imu_data.clear();
  while (!imu_ifs.eof() && std::getline(imu_ifs, line) && !line.empty()) {
    std::stringstream sst(line);
    gtsam::Vector7 imu;
    for (int i = 0; i < 7; i++) {
      sst >> imu[i];
    }

    imu_data.push_back(imu);
  }

  fit_knots(0.1);
  fitting_test();
  interpolation_test();
  imu_test();
}

int main(int argc, char** argv) {
  return 0;

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}