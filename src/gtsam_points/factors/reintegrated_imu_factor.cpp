// SPDX-License-Identifier: MIT
// Copyright (c) 2024  Kenji Koide (k.koide@aist.go.jp)
#include <gtsam_points/factors/reintegrated_imu_factor.hpp>

namespace gtsam_points {

ReintegratedImuMeasurements::ReintegratedImuMeasurements(
  const boost::shared_ptr<gtsam::PreintegrationParams>& p,
  const gtsam::imuBias::ConstantBias& biasHat)
: gtsam::PreintegratedImuMeasurements(p, biasHat) {}

ReintegratedImuMeasurements::~ReintegratedImuMeasurements() {}

void ReintegratedImuMeasurements::resetIntegration() {
  gtsam::PreintegratedImuMeasurements::resetIntegration();
  imu_data.clear();
}

void ReintegratedImuMeasurements::integrateMeasurement(const gtsam::Vector3& measuredAcc, const gtsam::Vector3& measuredOmega, double dt) {
  gtsam::PreintegratedImuMeasurements::integrateMeasurement(measuredAcc, measuredOmega, dt);
  gtsam::Vector7 imu;
  imu << measuredAcc, measuredOmega, dt;
  imu_data.emplace_back(imu);
}

const Eigen::Vector3d ReintegratedImuMeasurements::mean_acc() const {
  Eigen::Vector3d sum_acc = Eigen::Vector3d::Zero();
  for (const auto& imu : imu_data) {
    sum_acc += imu.block<3, 1>(0, 0);
  }
  return sum_acc / imu_data.size();
}

const Eigen::Vector3d ReintegratedImuMeasurements::mean_gyro() const {
  Eigen::Vector3d sum_gyro = Eigen::Vector3d::Zero();
  for (const auto& imu : imu_data) {
    sum_gyro += imu.block<3, 1>(3, 0);
  }
  return sum_gyro / imu_data.size();
}

ReintegratedImuFactor::ReintegratedImuFactor(
  gtsam::Key pose_i,
  gtsam::Key vel_i,
  gtsam::Key pose_j,
  gtsam::Key vel_j,
  gtsam::Key bias,
  const ReintegratedImuMeasurements& imu_measurements)
: gtsam::NonlinearFactor(gtsam::KeyVector{pose_i, vel_i, pose_j, vel_j, bias}),
  imu_measurements(imu_measurements) {}

ReintegratedImuFactor::~ReintegratedImuFactor() {}

void ReintegratedImuFactor::print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const {
  std::cout << s << "ReintegratedImuFactor";
  std::cout << "(" << keyFormatter(this->keys()[0]) << ", " << keyFormatter(this->keys()[1]) << ", " << keyFormatter(this->keys()[2]) << ", "
            << keyFormatter(this->keys()[3]) << ", " << keyFormatter(this->keys()[4]) << ")" << std::endl;
  std::cout << "|imu_data|=" << imu_measurements.imu_data.size() << std::endl;
}

boost::shared_ptr<gtsam::GaussianFactor> ReintegratedImuFactor::linearize(const gtsam::Values& values) const {
  imu_factor = create_imu_factor(values.at<gtsam::imuBias::ConstantBias>(keys()[4]));
  return imu_factor->linearize(values);
}

double ReintegratedImuFactor::error(const gtsam::Values& values) const {
  if (!imu_factor) {
    imu_factor = create_imu_factor(values.at<gtsam::imuBias::ConstantBias>(keys()[4]));
  }

  return imu_factor->error(values);
}

boost::shared_ptr<gtsam::ImuFactor> ReintegratedImuFactor::create_imu_factor(const gtsam::imuBias::ConstantBias& bias) const {
  gtsam::PreintegratedImuMeasurements pim(imu_measurements.params(), bias);
  for (const auto& imu : imu_measurements.imu_data) {
    const gtsam::Vector3 acc = imu.block<3, 1>(0, 0);
    const gtsam::Vector3 gyro = imu.block<3, 1>(3, 0);
    const double dt = imu[6];
    pim.integrateMeasurement(acc, gyro, dt);
  }

  return gtsam::make_shared<gtsam::ImuFactor>(keys()[0], keys()[1], keys()[2], keys()[3], keys()[4], pim);
}

}  // namespace gtsam_points