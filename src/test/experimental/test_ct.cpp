#include <fstream>
#include <iostream>

#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam_points/util/bspline.hpp>
#include <gtsam_points/util/continuous_trajectory.hpp>

#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>

int main(int argc, char** argv) {
  const std::string traj_path = "/home/koide/traj.txt";
  const std::string imu_path = "/home/koide/imu.txt";

  std::ifstream traj_ifs(traj_path);
  if (!traj_ifs) {
    std::cerr << "error: failed to open " << traj_path << std::endl;
    return 1;
  }

  std::vector<double> stamps;
  std::vector<gtsam::Pose3> poses;

  std::string line;
  while (!traj_ifs.eof() && std::getline(traj_ifs, line) && !line.empty()) {
    std::stringstream sst(line);

    double t;
    Eigen::Vector3d trans;
    Eigen::Quaterniond quat;

    sst >> t >> trans.x() >> trans.y() >> trans.z() >> quat.x() >> quat.y() >> quat.z() >> quat.w();

    stamps.push_back(t);
    poses.push_back(gtsam::Pose3(gtsam::Rot3(quat), trans));
  }

  std::ifstream imu_ifs(imu_path);
  if (!imu_ifs) {
    std::cerr << "error: failed to open " << imu_path << std::endl;
    return 1;
  }

  std::vector<Eigen::Matrix<double, 7, 1>> imu_data;

  using gtsam::symbol_shorthand::X;

  while (!imu_ifs.eof() && std::getline(imu_ifs, line) && !line.empty()) {
    std::stringstream sst(line);

    Eigen::Matrix<double, 7, 1> imu;
    for (int i = 0; i < 7; i++) {
      sst >> imu[i];
    }
    imu_data.push_back(imu);
  }

  gtsam_points::ContinuousTrajectory ct('x', stamps.front(), stamps.back(), 0.1);
  gtsam::Values values = ct.fit_knots(stamps, poses);

  for (int i = 0; i < imu_data.size(); i++) {
    const double t = imu_data[i][0];
    const Eigen::Vector3d a = imu_data[i].block<3, 1>(1, 0);
    const Eigen::Vector3d w = imu_data[i].block<3, 1>(4, 0);

    if (t < stamps.front() || t > stamps.back()) {
      continue;
    }

    const int knot_i = ct.knot_id(t);
    const double knot_t = ct.knot_stamp(knot_i);
    const double p = (t - knot_t) / ct.knot_interval;

    const Eigen::Vector3d g(0.0, 0.0, 9.80665);
    const gtsam::Vector6_ imu_ = gtsam_points::bspline_imu(X(knot_i), p, ct.knot_interval, g);
    const gtsam::Vector6 imu = imu_.value(values);

    std::cout << "sim:" << a.transpose() << " " << w.transpose() << std::endl;
    std::cout << "imu:" << imu.transpose() << std::endl;
  }

  auto viewer = guik::LightViewer::instance();

  for (int i = 0; i < stamps.size(); i += 25) {
    const Eigen::Affine3d model_matrix = Eigen::Isometry3d(poses[i].matrix()) * Eigen::UniformScaling<double>(0.3);
    viewer->update_drawable("keyframe_" + std::to_string(i), glk::Primitives::coordinate_system(), guik::VertexColor(model_matrix.cast<float>()));
  }

  for (double t = stamps.front(); t < stamps.back(); t += 0.1) {
    const gtsam::Pose3 pose = ct.pose(values, t);
    const Eigen::Affine3d model_matrix = Eigen::Isometry3d(pose.matrix()) * Eigen::UniformScaling<double>(0.1);
    viewer->update_drawable("pose_" + std::to_string(t), glk::Primitives::coordinate_system(), guik::VertexColor(model_matrix.cast<float>()));
  }

  viewer->spin();

  return 0;
}