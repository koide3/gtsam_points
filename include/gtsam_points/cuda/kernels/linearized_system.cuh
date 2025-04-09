// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <Eigen/Core>

namespace gtsam_points {

struct LinearizedSystem6 {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  __host__ void print() const {
    std::cout << "***" << std::endl;
    std::cout << "--- num_inliers ---" << std::endl << num_inliers << std::endl;
    std::cout << "--- error ---" << std::endl << error << std::endl;
    std::cout << "--- H_target ---" << std::endl << H_target << std::endl;
    std::cout << "--- H_source ---" << std::endl << H_source << std::endl;
    std::cout << "--- H_target_source ---" << std::endl << H_target_source << std::endl;
    std::cout << "--- b_target ---" << std::endl << b_target.transpose() << std::endl;
    std::cout << "--- b_source ---" << std::endl << b_source.transpose() << std::endl;
  }

  __host__ __device__ LinearizedSystem6 operator+(const LinearizedSystem6& rhs) const {
    LinearizedSystem6 sum;
    sum.num_inliers = num_inliers + rhs.num_inliers;
    sum.error = error + rhs.error;
    sum.H_target = H_target + rhs.H_target;
    sum.H_source = H_source + rhs.H_source;
    sum.H_target_source = H_target_source + rhs.H_target_source;
    sum.b_target = b_target + rhs.b_target;
    sum.b_source = b_source + rhs.b_source;

    return sum;
  }

  __host__ __device__ LinearizedSystem6& operator+=(const LinearizedSystem6& rhs) {
    num_inliers += rhs.num_inliers;
    error += rhs.error;
    H_target += rhs.H_target;
    H_source += rhs.H_source;
    H_target_source += rhs.H_target_source;
    b_target += rhs.b_target;
    b_source += rhs.b_source;

    return *this;
  }

  __host__ __device__ static LinearizedSystem6 zero() {
    LinearizedSystem6 system;
    system.num_inliers = 0;
    system.error = 0.0f;
    system.H_target.setZero();
    system.H_source.setZero();
    system.H_target_source.setZero();
    system.b_target.setZero();
    system.b_source.setZero();

    return system;
  }

public:
  int num_inliers;
  float error;
  Eigen::Matrix<float, 6, 6> H_target;
  Eigen::Matrix<float, 6, 6> H_source;
  Eigen::Matrix<float, 6, 6> H_target_source;
  Eigen::Matrix<float, 6, 1> b_target;
  Eigen::Matrix<float, 6, 1> b_source;
};

}  // namespace gtsam_points