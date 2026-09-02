// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>

namespace py = pybind11;
using namespace gtsam_points;

void define_pointcloud(py::module& m) {
  // PointCloud
  py::class_<PointCloud, std::shared_ptr<PointCloud>>(m, "PointCloud")
    .def(py::init<>())
    .def("size", &PointCloud::size)
    .def("has_times", &PointCloud::has_times)
    .def("has_points", &PointCloud::has_points)
    .def("has_normals", &PointCloud::has_normals)
    .def("has_covs", &PointCloud::has_covs)
    .def("has_intensities", &PointCloud::has_intensities)
    .def("check_times", &PointCloud::check_times)
    .def("check_points", &PointCloud::check_points)
    .def("check_normals", &PointCloud::check_normals)
    .def("check_covs", &PointCloud::check_covs)
    .def("check_intensities", &PointCloud::check_intensities)
    .def("save", &PointCloud::save, py::arg("path"))
    .def("save_compact", &PointCloud::save_compact, py::arg("path"))
    .def_property_readonly("num_points", [](const PointCloud& pc) { return pc.num_points; })
    .def_property_readonly(
      "points",
      [](const PointCloud& pc) {
        return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>>(pc.points[0].data(), pc.num_points, 4);
      })
    .def_property_readonly(
      "normals",
      [](const PointCloud& pc) {
        return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>>(pc.normals[0].data(), pc.num_points, 4);
      })
    .def_property_readonly(
      "covs",
      [](const PointCloud& pc) {
        return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 16, Eigen::RowMajor>>(pc.covs[0].data(), pc.num_points, 16);
      })
    .def_property_readonly("intensities", [](const PointCloud& pc) { return Eigen::Map<const Eigen::VectorXd>(pc.intensities, pc.num_points); })
    .def_property_readonly("times", [](const PointCloud& pc) { return Eigen::Map<const Eigen::VectorXd>(pc.times, pc.num_points); });

  // PointCloudCPU
  py::class_<PointCloudCPU, PointCloud, std::shared_ptr<PointCloudCPU>>(m, "PointCloudCPU")
    .def(py::init<>())
    .def("add_times", py::overload_cast<const std::vector<double>&>(&PointCloudCPU::add_times<double>), py::arg("times"))
    .def(
      "add_points",
      py::overload_cast<const std::vector<Eigen::Vector3d>&>(&PointCloudCPU::add_points<double, 3, std::allocator<Eigen::Vector3d>>),
      py::arg("points"))
    .def(
      "add_points",
      py::overload_cast<const std::vector<Eigen::Vector4d>&>(&PointCloudCPU::add_points<double, 4, std::allocator<Eigen::Vector4d>>),
      py::arg("points"))
    .def(
      "add_normals",
      py::overload_cast<const std::vector<Eigen::Vector3d>&>(&PointCloudCPU::add_normals<double, 3, std::allocator<Eigen::Vector3d>>),
      py::arg("normals"))
    .def(
      "add_normals",
      py::overload_cast<const std::vector<Eigen::Vector4d>&>(&PointCloudCPU::add_normals<double, 4, std::allocator<Eigen::Vector4d>>),
      py::arg("normals"))
    .def(
      "add_covs",
      py::overload_cast<const std::vector<Eigen::Matrix3d>&>(&PointCloudCPU::add_covs<double, 3, std::allocator<Eigen::Matrix3d>>),
      py::arg("covs"))
    .def(
      "add_covs",
      py::overload_cast<const std::vector<Eigen::Matrix4d>&>(&PointCloudCPU::add_covs<double, 4, std::allocator<Eigen::Matrix4d>>),
      py::arg("covs"))
    .def("add_intensities", py::overload_cast<const std::vector<double>&>(&PointCloudCPU::add_intensities<double>), py::arg("intensities"))
    .def_static("load", &PointCloudCPU::load, py::arg("path"));

  // functions
  m.def("sample", &gtsam_points::sample, py::arg("points"), py::arg("indices"), "Sample points by indices");
  m.def(
    "random_sampling",
    [](const PointCloudCPU::ConstPtr& points, double sampling_rate, std::uint64_t seed = 0) {
      std::mt19937 mt(seed ? seed : (Eigen::Vector4d(4537819, 3452, 453512, 12).dot(points->points[0])));
      return gtsam_points::random_sampling(points, sampling_rate, mt);
    },
    py::arg("points"),
    py::arg("sampling_rate"),
    py::arg("seed") = 0);
  m.def("voxelgrid_sampling", &gtsam_points::voxelgrid_sampling, py::arg("points"), py::arg("voxel_resolution"), py::arg("num_threads") = 1);
  m.def(
    "randomgrid_sampling",
    [](const PointCloudCPU::ConstPtr& points, double voxel_resolution, double sampling_rate, std::uint64_t seed = 0, int num_threads = 1) {
      std::mt19937 mt(seed ? seed : (Eigen::Vector4d(4537819, 3452, 453512, 12).dot(points->points[0])));
      return gtsam_points::randomgrid_sampling(points, voxel_resolution, sampling_rate, mt, num_threads);
    },
    py::arg("points"),
    py::arg("voxel_resolution"),
    py::arg("sampling_rate"),
    py::arg("seed") = 0,
    py::arg("num_threads") = 1);
  m.def("sort_by_time", &gtsam_points::sort_by_time, py::arg("points"));
  m.def(
    "transform",
    [](const PointCloud::ConstPtr& points, const Eigen::Matrix4d& transformation) {
      return gtsam_points::transform(points, Eigen::Isometry3d(transformation));
    },
    py::arg("points"),
    py::arg("transformation"));
  m.def(
    "transform_inplace",
    [](PointCloud& points, const Eigen::Matrix4d& transformation) {
      return gtsam_points::transform_inplace(points, Eigen::Isometry3d(transformation));
    },
    py::arg("points"),
    py::arg("transformation"));
  m.def("find_inlier_points", &gtsam_points::find_inlier_points, py::arg("points"), py::arg("neighbors"), py::arg("k"), py::arg("std_thresh") = 1.0);
  m.def(
    "remove_outliers",
    [](const PointCloud::ConstPtr& points, const std::vector<int>& neighbors, int k, double std_thresh = 1.0) {
      return gtsam_points::remove_outliers(points, neighbors, k, std_thresh);
    },
    py::arg("points"),
    py::arg("neighbors"),
    py::arg("k"),
    py::arg("std_thresh") = 1.0);
  m.def(
    "remove_outliers",
    [](const PointCloud::ConstPtr& points, int k, double std_thresh = 1.0, int num_threads = 1) {
      return gtsam_points::remove_outliers(points, k, std_thresh, num_threads);
    },
    py::arg("points"),
    py::arg("k") = 10,
    py::arg("std_thresh") = 1.0,
    py::arg("num_threads") = 1);
  m.def("distances", &gtsam_points::distances, py::arg("points"), py::arg("max_scan_count") = std::numeric_limits<size_t>::max());
  m.def("minmax_distance", &gtsam_points::minmax_distance, py::arg("points"), py::arg("max_scan_count") = std::numeric_limits<size_t>::max());
  m.def("median_distance", &gtsam_points::median_distance, py::arg("points"), py::arg("max_scan_count") = std::numeric_limits<size_t>::max());
  m.def(
    "merge_frames",
    [](const std::vector<Eigen::Matrix4d>& poses, const std::vector<PointCloud::ConstPtr>& frames, double downsample_resolution) {
      std::vector<Eigen::Isometry3d> poses_(poses.size());
      std::transform(poses.begin(), poses.end(), poses_.begin(), [](const Eigen::Matrix4d& pose) { return Eigen::Isometry3d(pose); });
      return gtsam_points::merge_frames(poses_, frames, downsample_resolution);
    },
    py::arg("poses"),
    py::arg("frames"),
    py::arg("downsample_resolution"));
}