#include <iostream>
#include <gtsam/geometry/Pose3.h>
#include <gtsam_points/util/expressions.hpp>


void eval_expr(const gtsam::Values& values) {
  gtsam::Pose3_ pose0(0);
  gtsam::Pose3_ pose1(1);

  const double t = 0.25;
  gtsam::Vector6_ vel = gtsam_points::logmap(gtsam::between(pose0, pose1));
  gtsam::Pose3_ inc = gtsam_points::expmap(t * vel);
  gtsam::Pose3_ pose = gtsam::compose(pose0, inc);

  gtsam::Point3_ pt(gtsam::Point3(1, 2, 3));
  gtsam::Point3_ transed = gtsam::transformFrom(pose, pt);

  const auto& expr = transed;

  std::vector<gtsam::Matrix> Hs(expr.keys().size());
  const auto val = expr.value(values, Hs);

  std::cout << "--- value ---" << std::endl << val << std::endl;
  for (int i = 0; i < Hs.size(); i++) {
    std::cout << "--- Hs[" << i << "] ---" << std::endl;
    std::cout << Hs[i] << std::endl;
  }
}

void eval_real(const gtsam::Values& values) {
  gtsam::Pose3 pose0 = values.at<gtsam::Pose3>(0);
  gtsam::Pose3 pose1 = values.at<gtsam::Pose3>(1);

  gtsam::Matrix6 H_delta_0, H_delta_1;
  gtsam::Pose3 delta = pose0.between(pose1, H_delta_0, H_delta_1);

  gtsam::Matrix6 H_vel_delta;
  gtsam::Vector6 vel = gtsam::Pose3::Logmap(delta, H_vel_delta);

  const double t = 0.25;
  gtsam::Matrix6 H_inc_vel;
  gtsam::Pose3 inc = gtsam::Pose3::Expmap(t * vel, H_inc_vel);

  gtsam::Matrix6 H_pose_0_a, H_pose_inc;
  gtsam::Pose3 pose = pose0.compose(inc, H_pose_0_a, H_pose_inc);

  gtsam::Matrix6 H_inc_0_b = H_pose_inc * H_inc_vel * t * H_vel_delta * H_delta_0;
  gtsam::Matrix6 H_pose_0 = H_pose_0_a + H_inc_0_b;
  gtsam::Matrix6 H_pose_1 = H_pose_inc * H_inc_vel * t * H_vel_delta * H_delta_1;

  gtsam::Matrix36 H_transed_pose;
  gtsam::Point3 pt(1, 2, 3);
  gtsam::Point3 transed = pose.transformFrom(pt, H_transed_pose);

  std::cout << "--- value ---" << std::endl << pt << std::endl;
  std::cout << "--- H1 ---" << std::endl << H_transed_pose * H_pose_0 << std::endl;
  std::cout << "--- H2 ---" << std::endl << H_transed_pose * H_pose_1 << std::endl;
}

int main(int argc, char** argv) {
  gtsam::Values values;
  values.insert(0, gtsam::Pose3::Expmap(gtsam::Vector6::Random()));
  values.insert(1, gtsam::Pose3::Expmap(gtsam::Vector6::Random()));

  eval_expr(values);
  eval_real(values);

  return 0;
}