// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#include "gtsam_points_python.hpp"

#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/segmentation/min_cut.hpp>
#include <gtsam_points/segmentation/region_growing.hpp>

using namespace gtsam_points;

void define_segmentation(py::module_& m) {
  // gtsam_points::RegionGrowingParams
  py::class_<RegionGrowingParams>(m, "RegionGrowingParams", "Region growing parameters")
    .def(py::init<>())
    .def_readwrite("distance_threshold", &RegionGrowingParams::distance_threshold, "Distance threshold")
    .def_readwrite("angle_threshold", &RegionGrowingParams::angle_threshold, "Angle threshold in radian")
    .def_readwrite("dilation_radius", &RegionGrowingParams::dilation_radius, "Radius of dilation after region growing")
    .def_readwrite("max_cluster_size", &RegionGrowingParams::max_cluster_size, "Maximum cluster size")
    .def_readwrite("max_steps", &RegionGrowingParams::max_steps, "Maximum number of update steps")
    .def_readwrite("num_threads", &RegionGrowingParams::num_threads, "Number of threads");

  m.def(
    "region_growing",
    [](const PointCloud::ConstPtr& points, const NearestNeighborSearch::ConstPtr& search, const Eigen::Vector3d& seed_point, const RegionGrowingParams& params) {
      if (!points->has_normals()) {
        throw std::invalid_argument("points must have normals (use estimate_normals() and add_normals())");
      }

      py::gil_scoped_release release;
      const Eigen::Vector4d seed(seed_point.x(), seed_point.y(), seed_point.z(), 1.0);
      auto context = region_growing_init(*points, *search, seed, params);
      while (!region_growing_update(context, *points, *search, params)) {
      }
      py::gil_scoped_acquire acquire;
      return convert_indices(context.cluster_indices);
    },
    py::arg("points"),
    py::arg("search"),
    py::arg("seed_point"),
    py::arg("params") = RegionGrowingParams(),
    "Extract a cluster of points connected to a seed point by region growing (points must have normals).\n"
    "Returns the indices of the cluster points.");

  // gtsam_points::MinCutParams
  py::class_<MinCutParams>(m, "MinCutParams", "Min-cut segmentation parameters")
    .def(py::init<>())
    .def_readwrite("distance_sigma", &MinCutParams::distance_sigma, "Distance sigma")
    .def_readwrite("angle_sigma", &MinCutParams::angle_sigma, "Angle sigma in radian")
    .def_readwrite("foreground_mask_radius", &MinCutParams::foreground_mask_radius, "Points within this radius from the source point are considered foreground")
    .def_readwrite("background_mask_radius", &MinCutParams::background_mask_radius, "Points out of this radius from the source point are considered background")
    .def_readwrite("foreground_weight", &MinCutParams::foreground_weight, "Weight for the foreground points")
    .def_readwrite("background_weight", &MinCutParams::background_weight, "Weight for the background points")
    .def_readwrite("k_neighbors", &MinCutParams::k_neighbors, "Number of neighbors")
    .def_readwrite("num_threads", &MinCutParams::num_threads, "Number of threads");

  // gtsam_points::MinCutResult
  py::class_<MinCutResult>(m, "MinCutResult", "Min-cut segmentation result")
    .def_readonly("source_index", &MinCutResult::source_index, "Source point index")
    .def_readonly("sink_index", &MinCutResult::sink_index, "Sink point index")
    .def_readonly("max_flow", &MinCutResult::max_flow, "Maximum flow")
    .def_property_readonly(
      "cluster_indices",
      [](const MinCutResult& result) { return convert_indices(result.cluster_indices); },
      "Indices of foreground points")
    .def("__repr__", [](const MinCutResult& result) {
      return "<gtsam_points.MinCutResult cluster_size=" + std::to_string(result.cluster_indices.size()) + ">";
    });

  m.def(
    "min_cut",
    [](const PointCloud::ConstPtr& points, const NearestNeighborSearch::ConstPtr& search, size_t source_pt_index, const MinCutParams& params) {
      if (!points->has_normals()) {
        throw std::invalid_argument("points must have normals (use estimate_normals() and add_normals())");
      }

      py::gil_scoped_release release;
      return min_cut(*points, *search, source_pt_index, params);
    },
    py::arg("points"),
    py::arg("search"),
    py::arg("source_pt_index"),
    py::arg("params") = MinCutParams(),
    "Min-cut segmentation with a source point given by index (points must have normals)");

  m.def(
    "min_cut",
    [](const PointCloud::ConstPtr& points, const NearestNeighborSearch::ConstPtr& search, const Eigen::Vector3d& source_pt, const MinCutParams& params) {
      if (!points->has_normals()) {
        throw std::invalid_argument("points must have normals (use estimate_normals() and add_normals())");
      }

      py::gil_scoped_release release;
      const Eigen::Vector4d source(source_pt.x(), source_pt.y(), source_pt.z(), 1.0);
      return min_cut(*points, *search, source, params);
    },
    py::arg("points"),
    py::arg("search"),
    py::arg("source_pt"),
    py::arg("params") = MinCutParams(),
    "Min-cut segmentation with a source point given by coordinates [3] (points must have normals)");
}
