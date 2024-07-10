#pragma once

#include <vector>
#include <Eigen/Core>

template <typename T, int D>
struct RandomSet {
  RandomSet()
  : num_points(128),
    points(num_points),
    normals(num_points),
    covs(num_points),
    intensities(num_points),
    times(num_points),
    aux1(num_points),
    aux2(num_points) {
    //
    for (int i = 0; i < num_points; i++) {
      points[i].setOnes();
      points[i].template head<3>() = Eigen::Matrix<T, 3, 1>::Random();
      normals[i].setZero();
      normals[i].template head<3>() = Eigen::Matrix<T, 3, 1>::Random().normalized();
      covs[i].setZero();
      covs[i].template block<3, 3>(0, 0) = Eigen::Matrix<T, 3, 3>::Random();
      covs[i] = (covs[i] * covs[i].transpose()).eval();
      intensities[i] = Eigen::Vector2d::Random()[0];
      times[i] = Eigen::Vector2d::Random()[0];

      aux1[i] = Eigen::Matrix<T, D, 1>::Random();
      aux2[i] = Eigen::Matrix<T, D, D>::Random();
    }
  }

  const int num_points;
  std::vector<Eigen::Matrix<T, D, 1>> points;
  std::vector<Eigen::Matrix<T, D, 1>> normals;
  std::vector<Eigen::Matrix<T, D, D>> covs;
  std::vector<T> intensities;
  std::vector<T> times;

  std::vector<Eigen::Matrix<T, D, 1>> aux1;
  std::vector<Eigen::Matrix<T, D, D>> aux2;
};
