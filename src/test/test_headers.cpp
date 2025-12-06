/**
 * Build and minimum run test for header only classes
 */
#include <gtest/gtest.h>
#include <gtsam_points/util/stopwatch.hpp>
#include <gtsam_points/util/easy_profiler.hpp>
#include <gtsam_points/util/runnning_statistics.hpp>
#include <gtsam_points/util/indexed_sliding_window.hpp>
#include <gtsam_points/factors/pose3_calib_factor.hpp>
#include <gtsam_points/factors/pose3_interpolation_factor.hpp>
#include <gtsam_points/factors/reintegrated_imu_factor.hpp>
#include <gtsam_points/factors/rotate_vector3_factor.hpp>

TEST(TestHeaders, IndexedSlidingWindow) {
  gtsam_points::IndexedSlidingWindow<std::shared_ptr<int>> window(true);
  window.emplace_back(std::make_shared<int>(10));
  window[0].reset();
}

TEST(TestHeaders, StopWatch) {
  gtsam_points::Stopwatch sw;
  sw.start();
  sw.stop();
}

TEST(TestHeaders, EasyProfiler) {
  std::stringstream sst;
  gtsam_points::EasyProfiler prof("prof", sst);
}

TEST(TestHeaders, RunningStatistics) {
  gtsam_points::RunningStatistics<double> stats;
  stats.add(1.0);
}

TEST(TestHeaders, Pose3CalibFactor) {
  gtsam_points::Pose3CalibFactor factor(0, 1, 2, nullptr);
}

TEST(TestHeaders, Pose3InterpolationFactor) {
  gtsam_points::Pose3InterpolationFactor factor(0, 1, 2, 3, nullptr);
}

TEST(TestHeaders, ReintegratedImuFactor) {
  gtsam_points::ReintegratedImuMeasurements measurements(gtsam::PreintegrationParams::MakeSharedU());
  gtsam_points::ReintegratedImuFactor factor(0, 1, 2, 3, 4, measurements);
}

TEST(TestHeaders, RotateVector3Factor) {
  gtsam_points::RotateVector3Factor factor(0, 1, gtsam::Vector3::Zero(), nullptr);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}