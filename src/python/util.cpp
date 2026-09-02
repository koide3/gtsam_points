// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#include "gtsam_points_python.hpp"

#include <gtsam_points/util/read_points.hpp>

using namespace gtsam_points;

void define_util(py::module_& m) {
  m.def(
    "read_points",
    [](const std::string& path) {
      const auto points = read_points(path);
      py::array_t<double> arr({static_cast<py::ssize_t>(points.size()), static_cast<py::ssize_t>(3)});
      auto view = arr.mutable_unchecked<2>();
      for (size_t i = 0; i < points.size(); i++) {
        view(i, 0) = points[i].x();
        view(i, 1) = points[i].y();
        view(i, 2) = points[i].z();
      }
      return arr;
    },
    py::arg("path"),
    "Read points [N, 3] from a binary file that contains a sequence of float32 (x, y, z) tuples");

  m.def(
    "read_points4",
    [](const std::string& path) {
      const auto points = read_points4(path);
      py::array_t<double> arr({static_cast<py::ssize_t>(points.size()), static_cast<py::ssize_t>(4)});
      auto view = arr.mutable_unchecked<2>();
      for (size_t i = 0; i < points.size(); i++) {
        view(i, 0) = points[i].x();
        view(i, 1) = points[i].y();
        view(i, 2) = points[i].z();
        view(i, 3) = points[i].w();
      }
      return arr;
    },
    py::arg("path"),
    "Read points [N, 4] from a binary file that contains a sequence of float32 (x, y, z, w) tuples (e.g., KITTI velodyne bin files)");

  m.def(
    "read_times",
    [](const std::string& path) {
      const auto times = read_times(path);
      py::array_t<double> arr(static_cast<py::ssize_t>(times.size()));
      auto view = arr.mutable_unchecked<1>();
      for (size_t i = 0; i < times.size(); i++) {
        view(i) = times[i];
      }
      return arr;
    },
    py::arg("path"),
    "Read per-point timestamps [N] from a binary file that contains a sequence of float32 values");
}
