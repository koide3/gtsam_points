// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#include <pybind11/pybind11.h>

namespace py = pybind11;

void define_factors(py::module& m);
void define_pointcloud(py::module& m);

PYBIND11_MODULE(gtsam_points, m) {
  m.doc() = "Efficient and parallel algorithms for point cloud registration";

  define_pointcloud(m);
  define_factors(m);
}