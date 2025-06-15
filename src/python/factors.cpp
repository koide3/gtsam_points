// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/factors/linear_damping_factor.hpp>

namespace py = pybind11;
using namespace gtsam_points;

void define_factors(py::module& m) {
  py::class_<LinearDampingFactor, gtsam::LinearContainerFactor, std::shared_ptr<LinearDampingFactor>>(m, "LinearDampingFactor")
    .def(py::init<>())
    .def(py::init<gtsam::Key, int, double>(), py::arg("key"), py::arg("dim"), py::arg("mu"));
}