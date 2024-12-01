// SPDX-License-Identifier: MIT
// Copyright (c) 2024  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam_points {

/**
 * @brief IMU measurements to be re-integrated.
 *       This class stores IMU measurements along with the pre-integrated IMU measurements.
 */
class ReintegratedImuMeasurements : public gtsam::PreintegratedImuMeasurements {
public:
  friend class ReintegratedImuFactor;

  ReintegratedImuMeasurements(
    const boost::shared_ptr<gtsam::PreintegrationParams>& p,
    const gtsam::imuBias::ConstantBias& biasHat = gtsam::imuBias::ConstantBias());
  ~ReintegratedImuMeasurements() override;

  void resetIntegration() override;
  void integrateMeasurement(const gtsam::Vector3& measuredAcc, const gtsam::Vector3& measuredOmega, double dt) override;

  const std::vector<gtsam::Vector7>& getImuData() const { return imu_data; }

private:
  std::vector<gtsam::Vector7> imu_data;  // [a, w, dt]
};

/**
 * @brief "Re"-integrated IMU factor (IMU factor without pre-integration).
 *        This factor re-integrates IMU measurements every linearization to better estimate the IMU bias.
 */
class ReintegratedImuFactor
: public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector3, gtsam::Pose3, gtsam::Vector3, gtsam::imuBias::ConstantBias, gtsam::Vector3> {
public:
  ReintegratedImuFactor(
    gtsam::Key pose_i,
    gtsam::Key vel_i,
    gtsam::Key pose_j,
    gtsam::Key vel_j,
    gtsam::Key bias,
    gtsam::Key gravity,
    const ReintegratedImuMeasurements& imu_measurements);
  ~ReintegratedImuFactor() override;

  size_t dim() const override { return 15; }

  gtsam::Vector evaluateError(
    const gtsam::Pose3& pose_i,
    const gtsam::Vector3& vel_i,
    const gtsam::Pose3& pose_j,
    const gtsam::Vector3& vel_j,
    const gtsam::imuBias::ConstantBias& bias,
    const gtsam::Vector3& gravity,
    boost::optional<gtsam::Matrix&> H_pose_i = boost::none,
    boost::optional<gtsam::Matrix&> H_vel_i = boost::none,
    boost::optional<gtsam::Matrix&> H_pose_j = boost::none,
    boost::optional<gtsam::Matrix&> H_vel_j = boost::none,
    boost::optional<gtsam::Matrix&> H_bias = boost::none,
    boost::optional<gtsam::Matrix&> H_gravity = boost::none) const override;

private:
  const ReintegratedImuMeasurements imu_measurements;
};

}  // namespace gtsam_points