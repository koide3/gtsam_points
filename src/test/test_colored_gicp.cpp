#include <random>
#include <iostream>

#include <gtest/gtest.h>

#include <gtsam/slam/PriorFactor.h>

#include <gtsam_ext/ann/kdtree.hpp>
#include <gtsam_ext/types/frame_cpu.hpp>
#include <gtsam_ext/factors/integrated_colored_gicp_factor.hpp>
#include <gtsam_ext/optimizers/levenberg_marquardt_ext.hpp>

struct ColoredGICPTestBase : public testing::Test {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  virtual void SetUp() override {
    std::mt19937 mt;
    std::normal_distribution<> ndist(0.0, 1e-2);

    Eigen::Isometry3d delta = Eigen::Isometry3d::Identity();
    delta.translation() += Eigen::Vector3d(0.15, 0.15, 0.25);
    delta.linear() = Eigen::AngleAxisd(0.05, Eigen::Vector3d(0.1, 0.2, 0.3).normalized()).toRotationMatrix();

    for (double x = -4.0; x <= 4.0; x += 0.02) {
      for (double y = -4.0; y < 4.0; y += 0.02) {
        const Eigen::Vector4d pt(x, y, 2.0, 1.0);

        target_points.push_back(pt + Eigen::Vector4d(ndist(mt), ndist(mt), ndist(mt), 0.0));

        const int dx = round(x);
        const int dy = round(y);
        const double d = Eigen::Vector2d(x - dx, y - dy).norm();
        const double intensity = d < 0.1 ? 1.0 : 0.0;
        target_intensities.push_back(intensity + ndist(mt));
      }
    }

    for (double x = -2.0; x <= 2.0; x += 0.02) {
      for (double y = -2.0; y < 2.0; y += 0.02) {
        const Eigen::Vector4d pt(x, y, 2.0, 1.0);

        source_points.push_back(delta * pt + Eigen::Vector4d(ndist(mt), ndist(mt), ndist(mt), 0.0));

        const int dx = round(x);
        const int dy = round(y);
        const double d = Eigen::Vector2d(x - dx, y - dy).norm();
        const double intensity = d < 0.1 ? 1.0 : 0.0;
        source_intensities.push_back(intensity + ndist(mt));
      }
    }
  }

  Eigen::Isometry3d delta;
  std::vector<double> target_intensities;
  std::vector<double> source_intensities;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> target_points;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> source_points;
};

int main(int argc, char** argv) {}