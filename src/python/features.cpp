// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#include "gtsam_points_python.hpp"

#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/features/normal_estimation.hpp>
#include <gtsam_points/features/covariance_estimation.hpp>
#include <gtsam_points/features/fpfh_estimation.hpp>

using namespace gtsam_points;

void define_features(py::module_& m) {
  m.def(
    "estimate_normals",
    [](const PointCloud::ConstPtr& points, int k_neighbors, int num_threads) {
      py::gil_scoped_release release;
      const auto normals = estimate_normals(*points, k_neighbors, num_threads);
      py::gil_scoped_acquire acquire;
      return convert_points(normals);
    },
    py::arg("points"),
    py::arg("k_neighbors") = 10,
    py::arg("num_threads") = 1,
    "Estimate point normals from neighboring points. Returns normals [N, 3].");

  m.def(
    "estimate_covariances",
    [](const PointCloud::ConstPtr& points, int k_neighbors, int num_threads) {
      py::gil_scoped_release release;
      const auto covs = estimate_covariances(*points, k_neighbors, num_threads);
      py::gil_scoped_acquire acquire;
      return convert_covs(covs.data(), covs.size());
    },
    py::arg("points"),
    py::arg("k_neighbors") = 10,
    py::arg("num_threads") = 1,
    "Estimate point covariances from neighboring points. Returns covariances [N, 4, 4].");

  // gtsam_points::FPFHEstimationParams
  py::class_<FPFHEstimationParams>(m, "FPFHEstimationParams", "PFH/FPFH estimation parameters")
    .def(py::init<>())
    .def_readwrite("search_radius", &FPFHEstimationParams::search_radius, "Neighbor search radius")
    .def_readwrite("max_num_neighbors", &FPFHEstimationParams::max_num_neighbors, "Maximum number of neighbors")
    .def_readwrite("num_threads", &FPFHEstimationParams::num_threads, "Number of threads");

  m.def(
    "estimate_fpfh",
    [](const PointCloud::ConstPtr& points, const NearestNeighborSearch::ConstPtr& search, const FPFHEstimationParams& params) {
      if (!points->has_normals()) {
        throw std::invalid_argument("points must have normals (use estimate_normals() and add_normals())");
      }

      py::gil_scoped_release release;
      NearestNeighborSearch::ConstPtr search_ = search;
      if (!search_) {
        search_ = std::make_shared<KdTree>(points->points, points->size());
      }
      const auto features = estimate_fpfh(*points, *search_, params);
      py::gil_scoped_acquire acquire;
      return convert_features(features);
    },
    py::arg("points"),
    py::arg("search") = nullptr,
    py::arg("params") = FPFHEstimationParams(),
    "Estimate FPFH features (points must have normals). Returns features [N, 33].");

  m.def(
    "estimate_pfh",
    [](const PointCloud::ConstPtr& points, const NearestNeighborSearch::ConstPtr& search, const PFHEstimationParams& params) {
      if (!points->has_normals()) {
        throw std::invalid_argument("points must have normals (use estimate_normals() and add_normals())");
      }

      py::gil_scoped_release release;
      NearestNeighborSearch::ConstPtr search_ = search;
      if (!search_) {
        search_ = std::make_shared<KdTree>(points->points, points->size());
      }
      const auto features = estimate_pfh(*points, *search_, params);
      py::gil_scoped_acquire acquire;
      return convert_features(features);
    },
    py::arg("points"),
    py::arg("search") = nullptr,
    py::arg("params") = PFHEstimationParams(),
    "Estimate PFH features (points must have normals). Returns features [N, 125].");
}
