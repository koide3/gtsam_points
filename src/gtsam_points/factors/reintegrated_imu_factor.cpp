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

ReintegratedImuFactor::ReintegratedImuFactor(
  gtsam::Key pose_i,
  gtsam::Key vel_i,
  gtsam::Key pose_j,
  gtsam::Key vel_j,
  gtsam::Key bias,
  gtsam::Key gravity,
  const ReintegratedImuMeasurements& imu_measurements)
: gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector3, gtsam::Pose3, gtsam::Vector3, gtsam::imuBias::ConstantBias, gtsam::Vector3>(
    gtsam::noiseModel::Isotropic::Sigma(15, 1.0),
    pose_i,
    vel_i,
    pose_j,
    vel_j,
    bias,
    gravity),
  imu_measurements(imu_measurements) {}

ReintegratedImuFactor::~ReintegratedImuFactor() {}

gtsam::Vector ReintegratedImuFactor::evaluateError(
  const gtsam::Pose3& pose_i,
  const gtsam::Vector3& vel_i,
  const gtsam::Pose3& pose_j,
  const gtsam::Vector3& vel_j,
  const gtsam::imuBias::ConstantBias& bias,
  const gtsam::Vector3& gravity,
  boost::optional<gtsam::Matrix&> H_pose_i,
  boost::optional<gtsam::Matrix&> H_vel_i,
  boost::optional<gtsam::Matrix&> H_pose_j,
  boost::optional<gtsam::Matrix&> H_vel_j,
  boost::optional<gtsam::Matrix&> H_bias,
  boost::optional<gtsam::Matrix&> H_gravity) const {
  //
  gtsam::NavState state_i(pose_i, vel_i);
  for (const auto& imu : imu_measurements.getImuData()) {
    const gtsam::Vector3 acc = bias.correctAccelerometer(imu.head<3>());
    const gtsam::Vector3 omega = bias.correctGyroscope(imu.segment<3>(3));
    const double dt = imu(6);

    gtsam::Matrix99 H_next_prev;
    gtsam::Matrix93 H_next_acc, H_next_omega;
    state_i = state_i.update(acc, omega, dt, H_next_prev, H_next_acc, H_next_omega);
  }
}

}  // namespace gtsam_points