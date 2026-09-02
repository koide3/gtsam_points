// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#include "gtsam_points_python.hpp"

#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/nonlinear/ISAM2Result.h>

#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_optimization_status.hpp>
#include <gtsam_points/optimizers/isam2_ext.hpp>
#include <gtsam_points/optimizers/isam2_result_ext.hpp>

using namespace gtsam_points;

void define_optimizers(py::module_& m) {
  // gtsam_points::LevenbergMarquardtOptimizationStatus
  py::class_<LevenbergMarquardtOptimizationStatus>(m, "LevenbergMarquardtOptimizationStatus", "Levenberg-Marquardt optimization status")
    .def_readonly("iterations", &LevenbergMarquardtOptimizationStatus::iterations, "Number of iterations")
    .def_readonly("total_inner_iterations", &LevenbergMarquardtOptimizationStatus::total_inner_iterations, "Number of LM lambda trials")
    .def_readonly("error", &LevenbergMarquardtOptimizationStatus::error, "Current error")
    .def_readonly("cost_change", &LevenbergMarquardtOptimizationStatus::cost_change, "Cost change")
    .def_readonly("lambda_", &LevenbergMarquardtOptimizationStatus::lambda, "Current lambda for LM")
    .def_readonly("elapsed_time", &LevenbergMarquardtOptimizationStatus::elapsed_time, "Time since optimization beginning")
    .def_readonly("linearization_time", &LevenbergMarquardtOptimizationStatus::linearization_time, "Time spent for linearization")
    .def_readonly("linear_solver_time", &LevenbergMarquardtOptimizationStatus::linear_solver_time, "Time spent for solving the linear system")
    .def("to_string", &LevenbergMarquardtOptimizationStatus::to_string, "Format the optimization status")
    .def("to_short_string", &LevenbergMarquardtOptimizationStatus::to_short_string, "Format the optimization status in a single line")
    .def("__repr__", &LevenbergMarquardtOptimizationStatus::to_short_string);

  // gtsam_points::LevenbergMarquardtExtParams
  py::class_<LevenbergMarquardtExtParams, gtsam::LevenbergMarquardtParams, std::shared_ptr<LevenbergMarquardtExtParams>>(
    m,
    "LevenbergMarquardtExtParams",
    "Levenberg-Marquardt optimizer parameters")
    .def(py::init<>())
    .def("set_verbose", &LevenbergMarquardtExtParams::set_verbose, "Print the optimization status to stdout for each iteration")
    .def(
      "set_callback",
      [](LevenbergMarquardtExtParams& self, const py::object& callback) {
        if (callback.is_none()) {
          self.callback = nullptr;
          return;
        }
        self.callback = [callback](const LevenbergMarquardtOptimizationStatus& status, const gtsam::Values& values) {
          py::gil_scoped_acquire acquire;
          callback(status, values);
        };
      },
      py::arg("callback"),
      "Set a callback called for each optimization iteration: callback(status, values)")
    .def(
      "set_termination_criteria",
      [](LevenbergMarquardtExtParams& self, const py::object& criteria) {
        if (criteria.is_none()) {
          self.termination_criteria = nullptr;
          return;
        }
        self.termination_criteria = [criteria](const gtsam::Values& values) {
          py::gil_scoped_acquire acquire;
          return criteria(values).cast<bool>();
        };
      },
      py::arg("criteria"),
      "Set a custom termination criteria: criteria(values) -> bool");

  // gtsam_points::LevenbergMarquardtOptimizerExt
  py::class_<LevenbergMarquardtOptimizerExt, std::shared_ptr<LevenbergMarquardtOptimizerExt>>(
    m,
    "LevenbergMarquardtOptimizerExt",
    "Levenberg-Marquardt optimizer with multi-threaded linearization support for gtsam_points factors")
    .def(
      py::init<const gtsam::NonlinearFactorGraph&, const gtsam::Values&, const LevenbergMarquardtExtParams&>(),
      py::arg("graph"),
      py::arg("initial_values"),
      py::arg("params") = LevenbergMarquardtExtParams(),
      "Create an optimizer for a nonlinear factor graph")
    .def(
      "optimize",
      [](LevenbergMarquardtOptimizerExt& self) {
        gtsam::Values values;
        {
          py::gil_scoped_release release;
          values = self.optimize();
        }
        return values;
      },
      "Optimize the factor graph and return the optimized values")
    .def("error", [](const LevenbergMarquardtOptimizerExt& self) { return self.error(); }, "Error at the current values")
    .def("iterations", [](const LevenbergMarquardtOptimizerExt& self) { return self.iterations(); }, "Number of iterations performed")
    .def("values", [](const LevenbergMarquardtOptimizerExt& self) { return self.values(); }, "Current values")
    .def("lambda_", &LevenbergMarquardtOptimizerExt::lambda, "Current damping value");

  // gtsam_points::ISAM2ResultExt
  py::class_<ISAM2ResultExt, gtsam::ISAM2Result, std::shared_ptr<ISAM2ResultExt>>(m, "ISAM2ResultExt", "ISAM2 update result")
    .def_readonly("delta", &ISAM2ResultExt::delta, "Maximum delta")
    .def_readonly("update_count", &ISAM2ResultExt::update_count, "Number of updates")
    .def_readonly("num_factors", &ISAM2ResultExt::num_factors, "Number of factors")
    .def_readonly("num_values", &ISAM2ResultExt::num_values, "Number of values")
    .def_readonly("elapsed_time", &ISAM2ResultExt::elapsed_time, "Elapsed time")
    .def("to_string", &ISAM2ResultExt::to_string, "Format the update result")
    .def("__repr__", &ISAM2ResultExt::to_string);

  // gtsam_points::ISAM2Ext
  py::class_<ISAM2Ext, std::shared_ptr<ISAM2Ext>>(m, "ISAM2Ext", "Incremental optimizer (ISAM2) with support for gtsam_points factors")
    .def(py::init<>(), "Create an ISAM2 instance with the default parameters")
    .def(py::init<const gtsam::ISAM2Params&>(), py::arg("params"), "Create an ISAM2 instance with the given parameters")
    .def(
      "update",
      [](ISAM2Ext& self, const gtsam::NonlinearFactorGraph& new_factors, const gtsam::Values& new_theta) {
        py::gil_scoped_release release;
        return self.update(new_factors, new_theta);
      },
      py::arg("new_factors") = gtsam::NonlinearFactorGraph(),
      py::arg("new_theta") = gtsam::Values(),
      "Add new factors and values, updating the solution and relinearizing as needed")
    .def(
      "calculate_estimate",
      [](ISAM2Ext& self) {
        py::gil_scoped_release release;
        return self.calculateEstimate();
      },
      "Compute the current best estimate of all variables")
    .def(
      "calculate_best_estimate",
      [](ISAM2Ext& self) {
        py::gil_scoped_release release;
        return self.calculateBestEstimate();
      },
      "Compute the full best estimate (backsubstitution without approximation)");
}
