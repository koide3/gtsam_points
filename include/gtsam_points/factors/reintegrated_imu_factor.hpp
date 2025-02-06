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

  const Eigen::Vector3d mean_acc() const;
  const Eigen::Vector3d mean_gyro() const;

public:
  std::vector<gtsam::Vector7> imu_data;  // [a, w, dt]
};
/**
 * @brief "Re"-integrated IMU factor (IMU factor without pre-integration).
 *        This factor re-integrates IMU measurements every linearization to better estimate the IMU bias.
 */
class ReintegratedImuFactor : public gtsam::NonlinearFactor {
public:
  ReintegratedImuFactor(
    gtsam::Key pose_i,
    gtsam::Key vel_i,
    gtsam::Key pose_j,
    gtsam::Key vel_j,
    gtsam::Key bias,
    const ReintegratedImuMeasurements& imu_measurements);
  ~ReintegratedImuFactor() override;

  size_t dim() const override { return 9; }

  virtual void print(const std::string& s = "", const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const override;

  boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& values) const override;
  double error(const gtsam::Values& values) const override;

  const ReintegratedImuMeasurements& measurements() const { return imu_measurements; }

private:
  boost::shared_ptr<gtsam::ImuFactor> create_imu_factor(const gtsam::imuBias::ConstantBias& bias) const;

private:
  const ReintegratedImuMeasurements imu_measurements;
  mutable boost::shared_ptr<gtsam::ImuFactor> imu_factor;
};
}  // namespace gtsam_points