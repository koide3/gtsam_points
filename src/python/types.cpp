// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#include "gtsam_points_python.hpp"

#include <random>

#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>

using namespace gtsam_points;

void define_types(py::module_& m) {
  // gtsam_points::PointCloud
  py::class_<PointCloud, std::shared_ptr<PointCloud>>(m, "PointCloud", "Point cloud that holds pointers to point attributes")
    .def("__repr__", [](const PointCloud& points) { return "<gtsam_points.PointCloud size=" + std::to_string(points.size()) + ">"; })
    .def("__len__", &PointCloud::size)
    .def("size", &PointCloud::size, "Number of points")
    .def("has_times", &PointCloud::has_times, "Check if the point cloud has per-point timestamps")
    .def("has_points", &PointCloud::has_points, "Check if the point cloud has points")
    .def("has_normals", &PointCloud::has_normals, "Check if the point cloud has point normals")
    .def("has_covs", &PointCloud::has_covs, "Check if the point cloud has point covariances")
    .def("has_intensities", &PointCloud::has_intensities, "Check if the point cloud has point intensities")
    .def_property_readonly(
      "points",
      [](const PointCloud& points) -> py::object {
        if (!points.has_points()) {
          return py::none();
        }
        return convert_points(points.points, points.size());
      },
      "Point coordinates [N, 3] (None if not available)")
    .def_property_readonly(
      "normals",
      [](const PointCloud& points) -> py::object {
        if (!points.has_normals()) {
          return py::none();
        }
        return convert_points(points.normals, points.size());
      },
      "Point normals [N, 3] (None if not available)")
    .def_property_readonly(
      "covs",
      [](const PointCloud& points) -> py::object {
        if (!points.has_covs()) {
          return py::none();
        }
        return convert_covs(points.covs, points.size());
      },
      "Point covariances [N, 4, 4] (None if not available)")
    .def_property_readonly(
      "times",
      [](const PointCloud& points) -> py::object {
        if (!points.has_times()) {
          return py::none();
        }
        return py::array_t<double>(points.size(), points.times);
      },
      "Per-point timestamps w.r.t. the first point [N] (None if not available)")
    .def_property_readonly(
      "intensities",
      [](const PointCloud& points) -> py::object {
        if (!points.has_intensities()) {
          return py::none();
        }
        return py::array_t<double>(points.size(), points.intensities);
      },
      "Point intensities [N] (None if not available)")
    .def("save", &PointCloud::save, py::arg("path"), "Save the point cloud data")
    .def("save_compact", &PointCloud::save_compact, py::arg("path"), "Save the point cloud data with a compact representation");

  // gtsam_points::PointCloudCPU
  py::class_<PointCloudCPU, PointCloud, std::shared_ptr<PointCloudCPU>>(m, "PointCloudCPU", "Point cloud frame on CPU memory")
    .def(py::init<>())
    .def(
      py::init([](const DoubleArray& points) {
        const auto converted = convert_points(points);
        return std::make_shared<PointCloudCPU>(converted);
      }),
      py::arg("points"),
      "Create a point cloud from a numpy array [N, 3] or [N, 4]")
    .def("__repr__", [](const PointCloudCPU& points) { return "<gtsam_points.PointCloudCPU size=" + std::to_string(points.size()) + ">"; })
    .def(
      "add_points",
      [](PointCloudCPU& self, const DoubleArray& points) {
        const auto converted = convert_points(points);
        self.add_points(converted);
      },
      py::arg("points"),
      "Add point coordinates [N, 3] or [N, 4]")
    .def(
      "add_times",
      [](PointCloudCPU& self, const DoubleArray& times) {
        const auto converted = convert_scalars(times, "times", self.size());
        self.add_times(converted);
      },
      py::arg("times"),
      "Add per-point timestamps [N]")
    .def(
      "add_normals",
      [](PointCloudCPU& self, const DoubleArray& normals) {
        const auto converted = convert_normals(normals);
        if (converted.size() != self.size()) {
          throw std::invalid_argument("normals must have the same number of elements as points");
        }
        self.add_normals(converted);
      },
      py::arg("normals"),
      "Add point normals [N, 3] or [N, 4]")
    .def(
      "add_covs",
      [](PointCloudCPU& self, const DoubleArray& covs) {
        const auto converted = convert_covs(covs);
        if (converted.size() != self.size()) {
          throw std::invalid_argument("covs must have the same number of elements as points");
        }
        self.add_covs(converted);
      },
      py::arg("covs"),
      "Add point covariances [N, 3, 3] or [N, 4, 4]")
    .def(
      "add_intensities",
      [](PointCloudCPU& self, const DoubleArray& intensities) {
        const auto converted = convert_scalars(intensities, "intensities", self.size());
        self.add_intensities(converted);
      },
      py::arg("intensities"),
      "Add point intensities [N]")
    .def("memory_usage", &PointCloudCPU::memory_usage, "Memory usage in bytes")
    .def_static("load", &PointCloudCPU::load, py::arg("path"), "Load point cloud data saved by PointCloud.save()")
    .def_static("clone", &PointCloudCPU::clone, py::arg("points"), "Deep copy a point cloud");

  // Point cloud utility functions
  m.def(
    "sample",
    [](const PointCloud::ConstPtr& points, const std::vector<int>& indices) { return sample(points, indices); },
    py::arg("points"),
    py::arg("indices"),
    "Sample points by indices");

  m.def(
    "random_sampling",
    [](const PointCloud::ConstPtr& points, double sampling_rate, std::uint64_t seed) {
      std::mt19937 mt(seed);
      py::gil_scoped_release release;
      return random_sampling(points, sampling_rate, mt);
    },
    py::arg("points"),
    py::arg("sampling_rate"),
    py::arg("seed") = 5489u,
    "Naive random sampling");

  m.def(
    "voxelgrid_sampling",
    [](const PointCloud::ConstPtr& points, double voxel_resolution, int num_threads) {
      py::gil_scoped_release release;
      return voxelgrid_sampling(points, voxel_resolution, num_threads);
    },
    py::arg("points"),
    py::arg("voxel_resolution"),
    py::arg("num_threads") = 1,
    "Voxel grid downsampling (takes the average of point attributes in each voxel)");

  m.def(
    "randomgrid_sampling",
    [](const PointCloud::ConstPtr& points, double voxel_resolution, double sampling_rate, std::uint64_t seed, int num_threads) {
      std::mt19937 mt(seed);
      py::gil_scoped_release release;
      return randomgrid_sampling(points, voxel_resolution, sampling_rate, mt, num_threads);
    },
    py::arg("points"),
    py::arg("voxel_resolution"),
    py::arg("sampling_rate"),
    py::arg("seed") = 5489u,
    py::arg("num_threads") = 1,
    "Voxel grid random sampling (spatially well-distributed random sampling without mixing point attributes)");

  m.def(
    "sort_by_time",
    [](const PointCloud::ConstPtr& points) {
      py::gil_scoped_release release;
      return sort_by_time(points);
    },
    py::arg("points"),
    "Sort points by time");

  m.def(
    "transform",
    [](const PointCloud::ConstPtr& points, const Eigen::Matrix4d& transformation) {
      const Eigen::Isometry3d T = convert_isometry(transformation);
      py::gil_scoped_release release;
      return transform(points, T);
    },
    py::arg("points"),
    py::arg("transformation"),
    "Transform points, normals, and covariances (transformation must be a 4x4 matrix)");

  m.def(
    "transform_inplace",
    [](const PointCloud::Ptr& points, const Eigen::Matrix4d& transformation) {
      const Eigen::Isometry3d T = convert_isometry(transformation);
      py::gil_scoped_release release;
      transform_inplace(points, T);
    },
    py::arg("points"),
    py::arg("transformation"),
    "Transform points, normals, and covariances inplace (transformation must be a 4x4 matrix)");

  m.def(
    "remove_outliers",
    [](const PointCloud::ConstPtr& points, int k, double std_thresh, int num_threads) {
      py::gil_scoped_release release;
      return remove_outliers(points, k, std_thresh, num_threads);
    },
    py::arg("points"),
    py::arg("k") = 10,
    py::arg("std_thresh") = 1.0,
    py::arg("num_threads") = 1,
    "Statistical outlier removal");

  m.def(
    "distances",
    [](const PointCloud::ConstPtr& points, size_t max_scan_count) {
      py::gil_scoped_release release;
      const auto dists = distances(points, max_scan_count);
      py::gil_scoped_acquire acquire;
      return py::array_t<double>(dists.size(), dists.data());
    },
    py::arg("points"),
    py::arg("max_scan_count") = std::numeric_limits<size_t>::max(),
    "Compute point distances from the origin");

  m.def(
    "minmax_distance",
    [](const PointCloud::ConstPtr& points, size_t max_scan_count) {
      py::gil_scoped_release release;
      return minmax_distance(points, max_scan_count);
    },
    py::arg("points"),
    py::arg("max_scan_count") = std::numeric_limits<size_t>::max(),
    "Compute min and max point distances from the origin");

  m.def(
    "median_distance",
    [](const PointCloud::ConstPtr& points, size_t max_scan_count) {
      py::gil_scoped_release release;
      return median_distance(points, max_scan_count);
    },
    py::arg("points"),
    py::arg("max_scan_count") = std::numeric_limits<size_t>::max(),
    "Compute median point distance from the origin");

  m.def(
    "merge_frames",
    [](const std::vector<Eigen::Matrix4d>& poses, const std::vector<PointCloud::ConstPtr>& frames, double downsample_resolution, py::object max_num_points) {
      std::vector<Eigen::Isometry3d> poses_(poses.size());
      std::transform(poses.begin(), poses.end(), poses_.begin(), [](const Eigen::Matrix4d& pose) { return convert_isometry(pose); });

      if (max_num_points.is_none()) {
        py::gil_scoped_release release;
        return merge_frames(poses_, frames, downsample_resolution);
      }

      const size_t max_num_points_ = max_num_points.cast<size_t>();
      py::gil_scoped_release release;
      return merge_frames(poses_, frames, downsample_resolution, max_num_points_);
    },
    py::arg("poses"),
    py::arg("frames"),
    py::arg("downsample_resolution"),
    py::arg("max_num_points") = py::none(),
    "Merge a set of frames into one frame (poses must be a list of 4x4 matrices)");

  // gtsam_points::GaussianVoxelMap
  py::class_<GaussianVoxelMap, std::shared_ptr<GaussianVoxelMap>>(m, "GaussianVoxelMap", "Gaussian distribution voxelmap")
    .def("voxel_resolution", &GaussianVoxelMap::voxel_resolution, "Voxel resolution")
    .def(
      "insert",
      [](GaussianVoxelMap& self, const PointCloud::ConstPtr& frame) {
        py::gil_scoped_release release;
        self.insert(*frame);
      },
      py::arg("frame"),
      "Insert a point cloud frame into the voxelmap")
    .def("save_compact", &GaussianVoxelMap::save_compact, py::arg("path"), "Save the voxelmap");

  // gtsam_points::GaussianVoxelMapCPU
  py::class_<GaussianVoxelMapCPU, GaussianVoxelMap, std::shared_ptr<GaussianVoxelMapCPU>>(m, "GaussianVoxelMapCPU", "Gaussian voxelmap on CPU memory")
    .def(py::init<double>(), py::arg("resolution"), "Create a Gaussian voxelmap with the given voxel resolution")
    .def("__repr__", [](const GaussianVoxelMapCPU& voxelmap) {
      return "<gtsam_points.GaussianVoxelMapCPU resolution=" + std::to_string(voxelmap.voxel_resolution()) + ">";
    })
    .def_static("load", &GaussianVoxelMapCPU::load, py::arg("path"), "Load a voxelmap saved by save_compact()");

  m.def(
    "overlap",
    [](const GaussianVoxelMap::ConstPtr& target, const PointCloud::ConstPtr& source, const Eigen::Matrix4d& T_target_source) {
      const Eigen::Isometry3d T = convert_isometry(T_target_source);
      py::gil_scoped_release release;
      return overlap(target, source, T);
    },
    py::arg("target"),
    py::arg("source"),
    py::arg("T_target_source"),
    "Calculate the fraction of source points that fall within the target voxelmap");

  m.def(
    "overlap",
    [](const std::vector<GaussianVoxelMap::ConstPtr>& targets, const PointCloud::ConstPtr& source, const std::vector<Eigen::Matrix4d>& Ts_target_source) {
      std::vector<Eigen::Isometry3d> Ts(Ts_target_source.size());
      std::transform(Ts_target_source.begin(), Ts_target_source.end(), Ts.begin(), [](const Eigen::Matrix4d& T) { return convert_isometry(T); });
      py::gil_scoped_release release;
      return overlap(targets, source, Ts);
    },
    py::arg("targets"),
    py::arg("source"),
    py::arg("Ts_target_source"),
    "Calculate the fraction of source points that fall within any of the target voxelmaps");
}
