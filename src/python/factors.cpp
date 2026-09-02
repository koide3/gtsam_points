// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#include "gtsam_points_python.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/types/gaussian_voxelmap.hpp>
#include <gtsam_points/ann/nearest_neighbor_search.hpp>
#include <gtsam_points/factors/integrated_icp_factor.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor.hpp>
#include <gtsam_points/factors/integrated_ct_icp_factor.hpp>
#include <gtsam_points/factors/integrated_ct_gicp_factor.hpp>

using namespace gtsam_points;

void define_factors(py::module_& m) {
  // gtsam_points::IntegratedMatchingCostFactor
  py::class_<IntegratedMatchingCostFactor, gtsam::NonlinearFactor, std::shared_ptr<IntegratedMatchingCostFactor>>(
    m,
    "IntegratedMatchingCostFactor",
    "Base class for scan matching factors that fully compute the matching cost and derivatives on linearization");

  // gtsam_points::IntegratedICPFactor
  py::class_<IntegratedICPFactor, IntegratedMatchingCostFactor, std::shared_ptr<IntegratedICPFactor>>(
    m,
    "IntegratedICPFactor",
    "Naive point-to-point ICP matching cost factor (Zhang, IJCV1994)")
    .def(
      py::init([](
                 gtsam::Key target_key,
                 gtsam::Key source_key,
                 const PointCloud::ConstPtr& target,
                 const PointCloud::ConstPtr& source,
                 const NearestNeighborSearch::ConstPtr& target_tree,
                 bool use_point_to_plane) {
        if (target_tree) {
          return std::make_shared<IntegratedICPFactor>(target_key, source_key, target, source, target_tree, use_point_to_plane);
        }
        return std::make_shared<IntegratedICPFactor>(target_key, source_key, target, source, use_point_to_plane);
      }),
      py::arg("target_key"),
      py::arg("source_key"),
      py::arg("target"),
      py::arg("source"),
      py::arg("target_tree") = nullptr,
      py::arg("use_point_to_plane") = false,
      "Create a binary ICP factor between target and source poses")
    .def(
      py::init([](
                 const gtsam::Pose3& fixed_target_pose,
                 gtsam::Key source_key,
                 const PointCloud::ConstPtr& target,
                 const PointCloud::ConstPtr& source,
                 const NearestNeighborSearch::ConstPtr& target_tree,
                 bool use_point_to_plane) {
        if (target_tree) {
          return std::make_shared<IntegratedICPFactor>(fixed_target_pose, source_key, target, source, target_tree, use_point_to_plane);
        }
        return std::make_shared<IntegratedICPFactor>(fixed_target_pose, source_key, target, source, use_point_to_plane);
      }),
      py::arg("fixed_target_pose"),
      py::arg("source_key"),
      py::arg("target"),
      py::arg("source"),
      py::arg("target_tree") = nullptr,
      py::arg("use_point_to_plane") = false,
      "Create a unary ICP factor between a fixed target pose and an active source pose")
    .def("set_num_threads", &IntegratedICPFactor::set_num_threads, py::arg("n"), "Set the number of threads used for linearization")
    .def(
      "set_max_correspondence_distance",
      &IntegratedICPFactor::set_max_correspondence_distance,
      py::arg("dist"),
      "Set the maximum distance between corresponding points")
    .def("set_point_to_plane_distance", &IntegratedICPFactor::set_point_to_plane_distance, py::arg("use"), "Enable or disable point-to-plane distance")
    .def(
      "set_correspondence_update_tolerance",
      &IntegratedICPFactor::set_correspondence_update_tolerance,
      py::arg("angle"),
      py::arg("trans"),
      "Set displacement thresholds to trigger the correspondence update");

  // gtsam_points::IntegratedPointToPlaneICPFactor
  py::class_<IntegratedPointToPlaneICPFactor, IntegratedICPFactor, std::shared_ptr<IntegratedPointToPlaneICPFactor>>(
    m,
    "IntegratedPointToPlaneICPFactor",
    "Point-to-plane ICP matching cost factor (target must have normals)")
    .def(
      py::init([](
                 gtsam::Key target_key,
                 gtsam::Key source_key,
                 const PointCloud::ConstPtr& target,
                 const PointCloud::ConstPtr& source,
                 const NearestNeighborSearch::ConstPtr& target_tree) {
        if (target_tree) {
          return std::make_shared<IntegratedPointToPlaneICPFactor>(target_key, source_key, target, source, target_tree);
        }
        return std::make_shared<IntegratedPointToPlaneICPFactor>(target_key, source_key, target, source);
      }),
      py::arg("target_key"),
      py::arg("source_key"),
      py::arg("target"),
      py::arg("source"),
      py::arg("target_tree") = nullptr,
      "Create a binary point-to-plane ICP factor between target and source poses");

  // gtsam_points::IntegratedGICPFactor
  py::class_<IntegratedGICPFactor, IntegratedMatchingCostFactor, std::shared_ptr<IntegratedGICPFactor>>(
    m,
    "IntegratedGICPFactor",
    "Generalized ICP matching cost factor (Segal et al., RSS2005). Target and source must have covariances.")
    .def(
      py::init([](
                 gtsam::Key target_key,
                 gtsam::Key source_key,
                 const PointCloud::ConstPtr& target,
                 const PointCloud::ConstPtr& source,
                 const NearestNeighborSearch::ConstPtr& target_tree) {
        if (target_tree) {
          return std::make_shared<IntegratedGICPFactor>(target_key, source_key, target, source, target_tree);
        }
        return std::make_shared<IntegratedGICPFactor>(target_key, source_key, target, source);
      }),
      py::arg("target_key"),
      py::arg("source_key"),
      py::arg("target"),
      py::arg("source"),
      py::arg("target_tree") = nullptr,
      "Create a binary GICP factor between target and source poses")
    .def(
      py::init([](
                 const gtsam::Pose3& fixed_target_pose,
                 gtsam::Key source_key,
                 const PointCloud::ConstPtr& target,
                 const PointCloud::ConstPtr& source,
                 const NearestNeighborSearch::ConstPtr& target_tree) {
        if (target_tree) {
          return std::make_shared<IntegratedGICPFactor>(fixed_target_pose, source_key, target, source, target_tree);
        }
        return std::make_shared<IntegratedGICPFactor>(fixed_target_pose, source_key, target, source);
      }),
      py::arg("fixed_target_pose"),
      py::arg("source_key"),
      py::arg("target"),
      py::arg("source"),
      py::arg("target_tree") = nullptr,
      "Create a unary GICP factor between a fixed target pose and an active source pose")
    .def("set_num_threads", &IntegratedGICPFactor::set_num_threads, py::arg("n"), "Set the number of threads used for linearization")
    .def(
      "set_max_correspondence_distance",
      &IntegratedGICPFactor::set_max_correspondence_distance,
      py::arg("dist"),
      "Set the maximum distance between corresponding points")
    .def(
      "set_correspondence_update_tolerance",
      &IntegratedGICPFactor::set_correspondence_update_tolerance,
      py::arg("angle"),
      py::arg("trans"),
      "Set displacement thresholds to trigger the correspondence update");

  // gtsam_points::IntegratedVGICPFactor
  py::class_<IntegratedVGICPFactor, IntegratedMatchingCostFactor, std::shared_ptr<IntegratedVGICPFactor>>(
    m,
    "IntegratedVGICPFactor",
    "Voxelized GICP matching cost factor (Koide et al., ICRA2021). Source must have covariances.")
    .def(
      py::init([](gtsam::Key target_key, gtsam::Key source_key, const GaussianVoxelMap::ConstPtr& target_voxels, const PointCloud::ConstPtr& source) {
        return std::make_shared<IntegratedVGICPFactor>(target_key, source_key, target_voxels, source);
      }),
      py::arg("target_key"),
      py::arg("source_key"),
      py::arg("target_voxels"),
      py::arg("source"),
      "Create a binary VGICP factor between target and source poses")
    .def(
      py::init([](const gtsam::Pose3& fixed_target_pose, gtsam::Key source_key, const GaussianVoxelMap::ConstPtr& target_voxels, const PointCloud::ConstPtr& source) {
        return std::make_shared<IntegratedVGICPFactor>(fixed_target_pose, source_key, target_voxels, source);
      }),
      py::arg("fixed_target_pose"),
      py::arg("source_key"),
      py::arg("target_voxels"),
      py::arg("source"),
      "Create a unary VGICP factor between a fixed target pose and an active source pose")
    .def("set_num_threads", &IntegratedVGICPFactor::set_num_threads, py::arg("n"), "Set the number of threads used for linearization");

  // gtsam_points::IntegratedCT_ICPFactor
  py::class_<IntegratedCT_ICPFactor, gtsam::NonlinearFactor, std::shared_ptr<IntegratedCT_ICPFactor>>(
    m,
    "IntegratedCT_ICPFactor",
    "Continuous-time ICP factor (Dellenbach et al., ICRA2022). Source must have per-point times.")
    .def(
      py::init([](
                 gtsam::Key source_t0_key,
                 gtsam::Key source_t1_key,
                 const PointCloud::ConstPtr& target,
                 const PointCloud::ConstPtr& source,
                 const NearestNeighborSearch::ConstPtr& target_tree) {
        if (target_tree) {
          return std::make_shared<IntegratedCT_ICPFactor>(source_t0_key, source_t1_key, target, source, target_tree);
        }
        return std::make_shared<IntegratedCT_ICPFactor>(source_t0_key, source_t1_key, target, source);
      }),
      py::arg("source_t0_key"),
      py::arg("source_t1_key"),
      py::arg("target"),
      py::arg("source"),
      py::arg("target_tree") = nullptr,
      "Create a continuous-time ICP factor between poses at the scan beginning (t0) and end (t1)")
    .def("set_num_threads", &IntegratedCT_ICPFactor::set_num_threads, py::arg("n"), "Set the number of threads used for linearization")
    .def(
      "set_max_correspondence_distance",
      &IntegratedCT_ICPFactor::set_max_correspondence_distance,
      py::arg("dist"),
      "Set the maximum distance between corresponding points")
    .def(
      "deskewed_source_points",
      [](IntegratedCT_ICPFactor& self, const gtsam::Values& values, bool local) {
        const auto points = self.deskewed_source_points(values, local);
        return convert_points(points);
      },
      py::arg("values"),
      py::arg("local") = false,
      "Compute deskewed source points [N, 3] with the estimated poses");

  // gtsam_points::IntegratedCT_GICPFactor
  py::class_<IntegratedCT_GICPFactor, IntegratedCT_ICPFactor, std::shared_ptr<IntegratedCT_GICPFactor>>(
    m,
    "IntegratedCT_GICPFactor",
    "Continuous-time GICP factor. Source must have per-point times, and target and source must have covariances.")
    .def(
      py::init([](
                 gtsam::Key source_t0_key,
                 gtsam::Key source_t1_key,
                 const PointCloud::ConstPtr& target,
                 const PointCloud::ConstPtr& source,
                 const NearestNeighborSearch::ConstPtr& target_tree) {
        if (target_tree) {
          return std::make_shared<IntegratedCT_GICPFactor>(source_t0_key, source_t1_key, target, source, target_tree);
        }
        return std::make_shared<IntegratedCT_GICPFactor>(source_t0_key, source_t1_key, target, source);
      }),
      py::arg("source_t0_key"),
      py::arg("source_t1_key"),
      py::arg("target"),
      py::arg("source"),
      py::arg("target_tree") = nullptr,
      "Create a continuous-time GICP factor between poses at the scan beginning (t0) and end (t1)");
}
