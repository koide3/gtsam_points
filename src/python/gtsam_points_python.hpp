// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <Eigen/Core>
#include <Eigen/Geometry>

#if PYBIND11_VERSION_MAJOR < 3
#error "gtsam_points python bindings require pybind11 >= 3.0 to interoperate with the GTSAM python bindings"
#endif

namespace py = pybind11;

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

/// @brief Validate and convert a numpy array [N, 3] or [N, 4] into homogeneous coordinates with the given w
inline std::vector<Eigen::Vector4d> convert_vectors(const DoubleArray& points, const std::string& name, double w) {
  if (points.ndim() != 2) {
    throw std::invalid_argument(name + " must be 2-dimensional (ndim=" + std::to_string(points.ndim()) + ")");
  }
  if (points.shape(1) != 3 && points.shape(1) != 4) {
    throw std::invalid_argument(name + " must have 3 or 4 columns (cols=" + std::to_string(points.shape(1)) + ")");
  }

  const py::ssize_t num_points = points.shape(0);
  const py::ssize_t cols = points.shape(1);
  const double* data = points.data();

  std::vector<Eigen::Vector4d> converted(num_points);
  for (py::ssize_t i = 0; i < num_points; i++) {
    converted[i] << data[i * cols], data[i * cols + 1], data[i * cols + 2], w;
  }
  return converted;
}

/// @brief Validate and convert a numpy array [N, 3] or [N, 4] into homogeneous points (w=1)
inline std::vector<Eigen::Vector4d> convert_points(const DoubleArray& points, const std::string& name = "points") {
  return convert_vectors(points, name, 1.0);
}

/// @brief Validate and convert a numpy array [N, 3] or [N, 4] into homogeneous normals (w=0)
inline std::vector<Eigen::Vector4d> convert_normals(const DoubleArray& normals, const std::string& name = "normals") {
  return convert_vectors(normals, name, 0.0);
}

/// @brief Convert homogeneous points into a numpy array [N, 3]
inline py::array_t<double> convert_points(const Eigen::Vector4d* points, const size_t num_points) {
  py::array_t<double> arr({static_cast<py::ssize_t>(num_points), static_cast<py::ssize_t>(3)});
  auto view = arr.mutable_unchecked<2>();
  for (size_t i = 0; i < num_points; i++) {
    view(i, 0) = points[i].x();
    view(i, 1) = points[i].y();
    view(i, 2) = points[i].z();
  }
  return arr;
}

inline py::array_t<double> convert_points(const std::vector<Eigen::Vector4d>& points) {
  return convert_points(points.data(), points.size());
}

/// @brief Validate and convert a numpy array [N, 3, 3] or [N, 4, 4] into 4x4 covariance matrices (cov(3, 3) = 0)
inline std::vector<Eigen::Matrix4d> convert_covs(const DoubleArray& covs, const std::string& name = "covs") {
  if (covs.ndim() != 3) {
    throw std::invalid_argument(name + " must be 3-dimensional [N, 3, 3] or [N, 4, 4] (ndim=" + std::to_string(covs.ndim()) + ")");
  }
  const py::ssize_t dim = covs.shape(1);
  if ((dim != 3 && dim != 4) || covs.shape(2) != dim) {
    throw std::invalid_argument(name + " must be [N, 3, 3] or [N, 4, 4]");
  }

  const py::ssize_t num_points = covs.shape(0);
  const double* data = covs.data();

  std::vector<Eigen::Matrix4d> converted(num_points);
  for (py::ssize_t i = 0; i < num_points; i++) {
    Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
    const double* c = data + i * dim * dim;
    for (py::ssize_t row = 0; row < dim; row++) {
      for (py::ssize_t col = 0; col < dim; col++) {
        cov(row, col) = c[row * dim + col];
      }
    }
    cov(3, 3) = 0.0;
    converted[i] = cov;
  }
  return converted;
}

/// @brief Convert 4x4 covariance matrices into a numpy array [N, 4, 4]
inline py::array_t<double> convert_covs(const Eigen::Matrix4d* covs, const size_t num_points) {
  py::array_t<double> arr({static_cast<py::ssize_t>(num_points), static_cast<py::ssize_t>(4), static_cast<py::ssize_t>(4)});
  auto view = arr.mutable_unchecked<3>();
  for (size_t i = 0; i < num_points; i++) {
    for (int row = 0; row < 4; row++) {
      for (int col = 0; col < 4; col++) {
        view(i, row, col) = covs[i](row, col);
      }
    }
  }
  return arr;
}

/// @brief Validate a numpy array [N] holding per-point scalar attributes (e.g., times and intensities)
inline std::vector<double> convert_scalars(const DoubleArray& values, const std::string& name, const py::ssize_t expected_size = -1) {
  if (values.ndim() != 1 && !(values.ndim() == 2 && values.shape(1) == 1)) {
    throw std::invalid_argument(name + " must be [N] or [N, 1] (ndim=" + std::to_string(values.ndim()) + ")");
  }
  if (expected_size >= 0 && values.shape(0) != expected_size) {
    throw std::invalid_argument(
      name + " must have the same number of elements as points (" + std::to_string(values.shape(0)) + " vs " + std::to_string(expected_size) + ")");
  }
  return std::vector<double>(values.data(), values.data() + values.shape(0));
}

/// @brief Validate and convert a numpy array [N, D] into a vector of D-dim feature vectors
inline std::vector<Eigen::VectorXd> convert_features(const DoubleArray& features, const std::string& name = "features") {
  if (features.ndim() != 2) {
    throw std::invalid_argument(name + " must be 2-dimensional [N, D] (ndim=" + std::to_string(features.ndim()) + ")");
  }

  const py::ssize_t num_features = features.shape(0);
  const py::ssize_t dim = features.shape(1);
  const double* data = features.data();

  std::vector<Eigen::VectorXd> converted(num_features);
  for (py::ssize_t i = 0; i < num_features; i++) {
    converted[i] = Eigen::Map<const Eigen::VectorXd>(data + i * dim, dim);
  }
  return converted;
}

/// @brief Convert a vector of D-dim feature vectors into a numpy array [N, D]
template <typename Feature>
inline py::array_t<double> convert_features(const std::vector<Feature>& features) {
  const py::ssize_t dim = features.empty() ? 0 : features.front().size();
  py::array_t<double> arr({static_cast<py::ssize_t>(features.size()), dim});
  auto view = arr.mutable_unchecked<2>();
  for (size_t i = 0; i < features.size(); i++) {
    for (py::ssize_t j = 0; j < dim; j++) {
      view(i, j) = features[i][j];
    }
  }
  return arr;
}

/// @brief Convert a 4x4 matrix into an isometry
inline Eigen::Isometry3d convert_isometry(const Eigen::Matrix4d& matrix) {
  return Eigen::Isometry3d(matrix);
}

/// @brief Convert a vector of indices into a numpy array [N]
template <typename T>
inline py::array_t<std::int64_t> convert_indices(const std::vector<T>& indices) {
  py::array_t<std::int64_t> arr(static_cast<py::ssize_t>(indices.size()));
  auto view = arr.mutable_unchecked<1>();
  for (size_t i = 0; i < indices.size(); i++) {
    view(i) = static_cast<std::int64_t>(indices[i]);
  }
  return arr;
}
