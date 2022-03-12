#include <iostream>

#include <gtsam/nonlinear/expressions.h>
#include <gtsam_ext/util/expressions.hpp>
#include <gtsam_ext/util/bspline.hpp>
#include <gtsam_ext/util/continuous_trajectory.hpp>

#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>

int main(int argc, char** argv) {
  auto viewer = guik::LightViewer::instance();

  gtsam::Values values;
  for (int i = 0; i < 10; i++) {
    const gtsam::Rot3 rot = gtsam::Rot3::Expmap(Eigen::Vector3d::Random() * 0.5);
    const gtsam::Vector3 trans = gtsam::Vector3(2 * i, std::cos(i * 0.5), std::sin(i * 0.1));
    const gtsam::Pose3 pose(rot, trans);
    values.insert(i, pose);

    viewer->update_drawable("knot_" + std::to_string(i), glk::Primitives::coordinate_system(), guik::VertexColor(pose.matrix().cast<float>()));
  }
  values.insert(1000, 0.1);

  const auto imu = gtsam_ext::bspline_imu(gtsam::Pose3_(0), gtsam::Pose3_(1), gtsam::Pose3_(2), gtsam::Pose3_(3), gtsam::Double_(gtsam::Key(1000)));
  std::cout << "imu:" << imu.value(values).transpose() << std::endl;

  const auto dr_t_ = gtsam_ext::bspline_angular_vel(
    gtsam::rotation(gtsam::Pose3_(0)),
    gtsam::rotation(gtsam::Pose3_(1)),
    gtsam::rotation(gtsam::Pose3_(2)),
    gtsam::rotation(gtsam::Pose3_(3)),
    gtsam::Double_(gtsam::Key(1000)));
  const auto dr_t = dr_t_.value(values);

  const auto dt_t2_ = gtsam_ext::bspline_linear_acc(
    gtsam_ext::translation(gtsam::Pose3_(0)),
    gtsam_ext::translation(gtsam::Pose3_(1)),
    gtsam_ext::translation(gtsam::Pose3_(2)),
    gtsam_ext::translation(gtsam::Pose3_(3)),
    gtsam::Double_(gtsam::Key(1000)));

  const auto dt_t2 = dt_t2_.value(values);

  std::cout << "dt_t2:" << dt_t2.transpose() << std::endl;
  std::cout << "dr_t:" << dr_t.transpose() << std::endl;

  return 0;
}