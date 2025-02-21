#include <random>
#include <gtest/gtest.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam_points/registration/alignment.hpp>

inline double sum_diffs(const Eigen::Isometry3d& T_target_source, const Eigen::Vector4d& target, const Eigen::Vector4d& source) {
  return (target - T_target_source * source).squaredNorm();
}

template <typename... Rest>
double sum_diffs(const Eigen::Isometry3d& T_target_source, const Eigen::Vector4d& target, const Eigen::Vector4d& source, const Rest&... rest) {
  return (target - T_target_source * source).squaredNorm() + sum_diffs(T_target_source, rest...);
}

template <typename... Args>
double sum_sq_errors(const Eigen::Isometry3d& T_target_source, const Args&... args) {
  static_assert(sizeof...(Args) % 2 == 0, "number of arguments must be even");
  return sum_diffs(T_target_source, args...);
}

template <int D, typename Evaluate, typename Retract>
bool is_optimal(const Eigen::Isometry3d& init, const Evaluate& evaluate, const Retract& retract, std::mt19937& mt) {
  const double error0 = evaluate(init);

  const std::vector<double> noise_scales = {0.01, 0.1, 1.0};
  for (int i = 0; i < 20; i++) {
    for (double noise_scale : noise_scales) {
      std::normal_distribution<> dist(0.0, noise_scale);
      Eigen::Matrix<double, D, 1> delta;
      std::generate(delta.data(), delta.data() + D, [&]() { return dist(mt); });
      const Eigen::Isometry3d T = init * retract(delta);
      const double error = evaluate(T);

      if (error < error0 - 1e-3) {
        return false;
      }
    }
  }

  return true;
}

TEST(AlignmentTest, AlignPoints_6DoF) {
  std::mt19937 mt;
  std::uniform_real_distribution<> udist(-1.0, 1.0);

  for (int i = 0; i < 100; i++) {
    std::vector<Eigen::Vector4d> target(3);
    std::vector<Eigen::Vector4d> source(3);
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> target_source_pairs(3);

    for (int j = 0; j < 3; j++) {
      target[j] = Eigen::Vector4d(udist(mt), udist(mt), udist(mt), 1.0);
      source[j] = Eigen::Vector4d(udist(mt), udist(mt), udist(mt), 1.0);
      target_source_pairs[j] = {target[j].head<3>(), source[j].head<3>()};
    }

    const auto T_target_source = gtsam_points::align_points_se3(target[0], target[1], target[2], source[0], source[1], source[2]);
    const auto gt_T_target_source = gtsam::Pose3::Align(target_source_pairs).get();

    EXPECT_NEAR((T_target_source.matrix() - gt_T_target_source.matrix()).array().abs().maxCoeff(), 0.0, 1e-3);

    const auto evaluate = [&](const Eigen::Isometry3d& T) {
      return sum_sq_errors(T, target[0], source[0], target[1], source[1], target[2], source[2]);
    };
    const auto retract = [&](const Eigen::Matrix<double, 6, 1>& delta) {
      Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
      T.linear() = Eigen::AngleAxisd(delta.head<3>().norm(), delta.head<3>().normalized()).toRotationMatrix();
      T.translation() = delta.tail<3>();
      return T;
    };

    EXPECT_EQ(is_optimal<6>(T_target_source, evaluate, retract, mt), true);
  }
}

TEST(AlignmentTest, AlignPoints_4DoF) {
  std::mt19937 mt;
  std::uniform_real_distribution<> udist(-1.0, 1.0);

  for (int i = 0; i < 100; i++) {
    std::vector<Eigen::Vector4d> target(2);
    std::vector<Eigen::Vector4d> source(2);

    for (int j = 0; j < 2; j++) {
      target[j] = Eigen::Vector4d(udist(mt), udist(mt), udist(mt), 1.0);
      source[j] = Eigen::Vector4d(udist(mt), udist(mt), udist(mt), 1.0);
    }

    const auto T_target_source = gtsam_points::align_points_4dof(target[0], target[1], source[0], source[1]);

    const Eigen::Matrix3d R = T_target_source.linear();
    EXPECT_NEAR(R.col(0).norm(), 1.0, 1e-6);
    EXPECT_NEAR(R.col(1).norm(), 1.0, 1e-6);
    EXPECT_NEAR(R.col(2).norm(), 1.0, 1e-6);
    EXPECT_NEAR(R.col(0).dot(R.col(1)), 0.0, 1e-6);
    EXPECT_NEAR(R.col(1).dot(R.col(2)), 0.0, 1e-6);
    EXPECT_NEAR((R.col(0).cross(R.col(1)) - R.col(2)).cwiseAbs().maxCoeff(), 0.0, 1e-6);
    EXPECT_NEAR((R.col(2) - Eigen::Vector3d::UnitZ()).cwiseAbs().maxCoeff(), 0.0, 1e-6);

    const auto evaluate = [&](const Eigen::Isometry3d& T) { return sum_sq_errors(T, target[0], source[0], target[1], source[1]); };
    const auto retract = [&](const Eigen::Matrix<double, 4, 1>& delta) {
      Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
      T.linear().topLeftCorner<2, 2>() = Eigen::Rotation2Dd(delta(0)).toRotationMatrix();
      T.translation() = delta.tail<3>();
      return T;
    };

    EXPECT_EQ(is_optimal<4>(T_target_source, evaluate, retract, mt), true);
  }
}

TEST(AlignmentTest, AlignPointSets_6DoF) {
  std::mt19937 mt;
  std::vector<int> num_samples = {3, 5, 10, 100};

  std::uniform_real_distribution<> wdist(1e-3, 1.0);
  std::uniform_real_distribution<> udist(-1.0, 1.0);

  for (int i = 0; i < 100; i++) {
    for (int N : num_samples) {
      std::vector<double> weights(N);
      std::vector<Eigen::Vector4d> target(N);
      std::vector<Eigen::Vector4d> source(N);

      for (int j = 0; j < N; j++) {
        weights[j] = wdist(mt);
        target[j] = Eigen::Vector4d(udist(mt), udist(mt), udist(mt), 1.0);
        source[j] = Eigen::Vector4d(udist(mt), udist(mt), udist(mt), 1.0);
      }

      const auto T_target_source = gtsam_points::align_points_se3(target.data(), source.data(), weights.data(), N);

      const Eigen::Matrix3d R = T_target_source.linear();
      EXPECT_NEAR(R.col(0).norm(), 1.0, 1e-6);
      EXPECT_NEAR(R.col(1).norm(), 1.0, 1e-6);
      EXPECT_NEAR(R.col(2).norm(), 1.0, 1e-6);
      EXPECT_NEAR(R.col(0).dot(R.col(1)), 0.0, 1e-6);
      EXPECT_NEAR(R.col(1).dot(R.col(2)), 0.0, 1e-6);
      EXPECT_NEAR((R.col(0).cross(R.col(1)) - R.col(2)).cwiseAbs().maxCoeff(), 0.0, 1e-6);

      const auto evaluate = [&](const Eigen::Isometry3d& T) {
        double sum_errors = 0.0;
        for (int j = 0; j < N; j++) {
          sum_errors += weights[j] * (target[j] - T * source[j]).squaredNorm();
        }
        return sum_errors;
      };
      const auto retract = [&](const Eigen::Matrix<double, 6, 1>& delta) {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.linear() = Eigen::AngleAxisd(delta.head<3>().norm(), delta.head<3>().normalized()).toRotationMatrix();
        T.translation() = delta.tail<3>();
        return T;
      };

      EXPECT_EQ(is_optimal<6>(T_target_source, evaluate, retract, mt), true);
    }
  }
}

TEST(AlignmentTest, AlignPointSets_4DoF) {
  std::mt19937 mt;
  std::vector<int> num_samples = {3, 5, 10, 100};

  std::uniform_real_distribution<> wdist(1e-3, 1.0);
  std::uniform_real_distribution<> udist(-1.0, 1.0);

  for (int i = 0; i < 100; i++) {
    for (int N : num_samples) {
      std::vector<double> weights(N);
      std::vector<Eigen::Vector4d> target(N);
      std::vector<Eigen::Vector4d> source(N);

      for (int j = 0; j < N; j++) {
        weights[j] = wdist(mt);
        target[j] = Eigen::Vector4d(udist(mt), udist(mt), udist(mt), 1.0);
        source[j] = Eigen::Vector4d(udist(mt), udist(mt), udist(mt), 1.0);
      }

      const auto T_target_source = gtsam_points::align_points_4dof(target.data(), source.data(), weights.data(), N);

      const Eigen::Matrix3d R = T_target_source.linear();
      EXPECT_NEAR(R.col(0).norm(), 1.0, 1e-6);
      EXPECT_NEAR(R.col(1).norm(), 1.0, 1e-6);
      EXPECT_NEAR(R.col(2).norm(), 1.0, 1e-6);
      EXPECT_NEAR(R.col(0).dot(R.col(1)), 0.0, 1e-6);
      EXPECT_NEAR(R.col(1).dot(R.col(2)), 0.0, 1e-6);
      EXPECT_NEAR((R.col(0).cross(R.col(1)) - R.col(2)).cwiseAbs().maxCoeff(), 0.0, 1e-6);
      EXPECT_NEAR((R.col(2) - Eigen::Vector3d::UnitZ()).cwiseAbs().maxCoeff(), 0.0, 1e-6);

      const auto evaluate = [&](const Eigen::Isometry3d& T) {
        double sum_errors = 0.0;
        for (int j = 0; j < N; j++) {
          sum_errors += weights[j] * (target[j] - T * source[j]).squaredNorm();
        }
        return sum_errors;
      };
      const auto retract = [&](const Eigen::Matrix<double, 4, 1>& delta) {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.linear().topLeftCorner<2, 2>() = Eigen::Rotation2Dd(delta(0)).toRotationMatrix();
        T.translation() = delta.tail<3>();
        return T;
      };

      EXPECT_EQ(is_optimal<4>(T_target_source, evaluate, retract, mt), true);
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}