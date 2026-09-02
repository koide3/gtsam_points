// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#include <pybind11/pybind11.h>

namespace py = pybind11;

void define_types(py::module_& m);
void define_ann(py::module_& m);
void define_features(py::module_& m);
void define_registration(py::module_& m);
void define_segmentation(py::module_& m);
void define_factors(py::module_& m);
void define_optimizers(py::module_& m);
void define_util(py::module_& m);

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(gtsam_points, m) {
  m.doc() = "A collection of GTSAM factors and optimizers for range-based SLAM";

  // The GTSAM python module must be loaded first so that gtsam types (e.g., gtsam.Values and
  // gtsam.NonlinearFactor) are registered and can be shared with this module.
  try {
    py::module_::import("gtsam");
  } catch (py::error_already_set& e) {
    py::raise_from(e, PyExc_ImportError, "gtsam_points requires the GTSAM python bindings (https://github.com/borglab/gtsam)");
    throw py::error_already_set();
  }

  define_util(m);
  define_types(m);
  define_ann(m);
  define_features(m);
  define_registration(m);
  define_segmentation(m);
  define_factors(m);
  define_optimizers(m);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
