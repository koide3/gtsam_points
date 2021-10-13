#include <iostream>

#include <gtsam/slam/expressions.h>
#include <gtsam/nonlinear/Values.h>

void expr(const gtsam::Values& values, const Eigen::Vector3d& point) {
  gtsam::Pose3_ pose0(0);
  gtsam::Pose3_ pose1(1);

  gtsam::Point3_ pt(point);
  gtsam::Point3_ transed = gtsam::transformFrom(gtsam::between(pose0, pose1), pt);

  std::vector<gtsam::Matrix> Hs(transed.keys().size());
  auto val = transed.value(values, Hs);

  std::cout << "--- val ---" << std::endl << val << std::endl;
  for (int i = 0; i < Hs.size(); i++) {
    std::cout << "--- H[" << i << "] ---" << std::endl << Hs[i] << std::endl;
  }
}

void hand(const gtsam::Values& values, const Eigen::Vector3d& point) {
  gtsam::Pose3 pose0 = values.at<gtsam::Pose3>(0);
  gtsam::Pose3 pose1 = values.at<gtsam::Pose3>(1);

  gtsam::Pose3 delta = pose0.inverse() * pose1;
  gtsam::Vector3 pt = delta * point;

  std::cout << "--- pt ---" << std::endl << pt << std::endl;

  gtsam::Matrix36 H0 = gtsam::Matrix36::Zero();
  H0.block<3, 3>(0, 0) = gtsam::SO3::Hat(pt);
  H0.block<3, 3>(0, 3) = -gtsam::Matrix3::Identity();

  gtsam::Matrix36 H1 = gtsam::Matrix36::Zero();
  H1.block<3, 3>(0, 0) = -delta.rotation().matrix() * gtsam::SO3::Hat(point);
  H1.block<3, 3>(0, 3) = delta.rotation().matrix();

  std::cout << "--- H0 ---" << std::endl << H0 << std::endl;
  std::cout << "--- H1 ---" << std::endl << H1 << std::endl;
}

int main(int argc, char** argv) {
  gtsam::Values values;
  values.insert(0, gtsam::Pose3::Expmap(gtsam::Vector6::Random()));
  values.insert(1, gtsam::Pose3::Expmap(gtsam::Vector6::Random()));

  Eigen::Vector3d pt = Eigen::Vector3d::Random();

  expr(values, pt);
  hand(values, pt);

  return 0;
}