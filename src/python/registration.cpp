// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#include "gtsam_points_python.hpp"

#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/ann/kdtreex.hpp>
#include <gtsam_points/registration/ransac.hpp>
#include <gtsam_points/registration/graduated_non_convexity.hpp>

using namespace gtsam_points;

namespace {

/// @brief Build a KdTreeX over feature vectors
std::shared_ptr<KdTreeX<-1>> build_features_tree(const std::vector<Eigen::VectorXd>& features) {
  return std::make_shared<KdTreeX<-1>>(features.data(), features.size());
}

}  // namespace

void define_registration(py::module_& m) {
  // gtsam_points::RegistrationResult
  py::class_<RegistrationResult>(m, "RegistrationResult", "Registration result")
    .def_readonly("inlier_rate", &RegistrationResult::inlier_rate, "Inlier rate")
    .def_property_readonly(
      "T_target_source",
      [](const RegistrationResult& result) -> Eigen::Matrix4d { return result.T_target_source.matrix(); },
      "Estimated transformation (4x4 matrix)")
    .def("__repr__", [](const RegistrationResult& result) {
      return "<gtsam_points.RegistrationResult inlier_rate=" + std::to_string(result.inlier_rate) + ">";
    });

  // gtsam_points::RANSACParams
  py::class_<RANSACParams>(m, "RANSACParams", "RANSAC parameters")
    .def(py::init<>())
    .def_readwrite("max_iterations", &RANSACParams::max_iterations, "Maximum number of iterations")
    .def_readwrite("early_stop_inlier_rate", &RANSACParams::early_stop_inlier_rate, "Inlier rate for early stopping")
    .def_readwrite("poly_error_thresh", &RANSACParams::poly_error_thresh, "Polynomial error threshold")
    .def_readwrite("inlier_voxel_resolution", &RANSACParams::inlier_voxel_resolution, "Inlier voxel resolution")
    .def_readwrite("dof", &RANSACParams::dof, "Degrees of freedom (must be 6 (SE3) or 4 (XYZ+RZ))")
    .def_readwrite("seed", &RANSACParams::seed, "Random seed")
    .def_readwrite("num_threads", &RANSACParams::num_threads, "Number of threads");

  // gtsam_points::GNCParams
  py::class_<GNCParams>(m, "GNCParams", "Graduated non-convexity parameters")
    .def(py::init<>())
    .def_readwrite("max_init_samples", &GNCParams::max_init_samples, "Maximum number of initial samples")
    .def_readwrite("reciprocal_check", &GNCParams::reciprocal_check, "Reciprocal check")
    .def_readwrite("tuple_check", &GNCParams::tuple_check, "Length similarity check")
    .def_readwrite("tuple_thresh", &GNCParams::tuple_thresh, "Length similarity threshold")
    .def_readwrite("max_num_tuples", &GNCParams::max_num_tuples, "Number of tuples to be sampled")
    .def_readwrite("div_factor", &GNCParams::div_factor, "Division factor for graduated non-convexity")
    .def_readwrite("max_corr_dist", &GNCParams::max_corr_dist, "Maximum correspondence distance")
    .def_readwrite("innter_iterations", &GNCParams::innter_iterations, "Number of inner iterations")
    .def_readwrite("max_iterations", &GNCParams::max_iterations, "Maximum number of iterations")
    .def_readwrite("dof", &GNCParams::dof, "Degrees of freedom (must be 6 (SE3) or 4 (XYZ+RZ))")
    .def_readwrite("verbose", &GNCParams::verbose, "Verbose mode")
    .def_readwrite("seed", &GNCParams::seed, "Random seed")
    .def_readwrite("num_threads", &GNCParams::num_threads, "Number of threads");

  m.def(
    "estimate_pose_ransac",
    [](
      const PointCloud::ConstPtr& target,
      const PointCloud::ConstPtr& source,
      const DoubleArray& target_features,
      const DoubleArray& source_features,
      const KdTree::ConstPtr& target_tree,
      const NearestNeighborSearch::ConstPtr& target_features_tree,
      const RANSACParams& params) {
      const auto target_features_ = convert_features(target_features, "target_features");
      const auto source_features_ = convert_features(source_features, "source_features");
      if (target_features_.size() != target->size() || source_features_.size() != source->size()) {
        throw std::invalid_argument("features must have the same number of elements as points");
      }

      py::gil_scoped_release release;

      NearestNeighborSearch::ConstPtr target_tree_ = target_tree;
      if (!target_tree_) {
        target_tree_ = std::make_shared<KdTree>(target->points, target->size());
      }

      NearestNeighborSearch::ConstPtr target_features_tree_ = target_features_tree;
      if (!target_features_tree_) {
        target_features_tree_ = build_features_tree(target_features_);
      }

      return estimate_pose_ransac(*target, *source, target_features_.data(), source_features_.data(), *target_tree_, *target_features_tree_, params);
    },
    py::arg("target"),
    py::arg("source"),
    py::arg("target_features"),
    py::arg("source_features"),
    py::arg("target_tree") = nullptr,
    py::arg("target_features_tree") = nullptr,
    py::arg("params") = RANSACParams(),
    "Estimate the transformation between two point clouds using feature-based RANSAC.\n"
    "Features must be [N, D] arrays (e.g., FPFH features from estimate_fpfh()).");

  m.def(
    "estimate_pose_gnc",
    [](
      const PointCloud::ConstPtr& target,
      const PointCloud::ConstPtr& source,
      const DoubleArray& target_features,
      const DoubleArray& source_features,
      const KdTree::ConstPtr& target_tree,
      const NearestNeighborSearch::ConstPtr& target_features_tree,
      const NearestNeighborSearch::ConstPtr& source_features_tree,
      const GNCParams& params) {
      const auto target_features_ = convert_features(target_features, "target_features");
      const auto source_features_ = convert_features(source_features, "source_features");
      if (target_features_.size() != target->size() || source_features_.size() != source->size()) {
        throw std::invalid_argument("features must have the same number of elements as points");
      }

      py::gil_scoped_release release;

      NearestNeighborSearch::ConstPtr target_tree_ = target_tree;
      if (!target_tree_) {
        target_tree_ = std::make_shared<KdTree>(target->points, target->size());
      }

      NearestNeighborSearch::ConstPtr target_features_tree_ = target_features_tree;
      if (!target_features_tree_) {
        target_features_tree_ = build_features_tree(target_features_);
      }

      NearestNeighborSearch::ConstPtr source_features_tree_ = source_features_tree;
      if (!source_features_tree_) {
        source_features_tree_ = build_features_tree(source_features_);
      }

      return estimate_pose_gnc(
        *target,
        *source,
        target_features_.data(),
        source_features_.data(),
        *target_tree_,
        *target_features_tree_,
        *source_features_tree_,
        params);
    },
    py::arg("target"),
    py::arg("source"),
    py::arg("target_features"),
    py::arg("source_features"),
    py::arg("target_tree") = nullptr,
    py::arg("target_features_tree") = nullptr,
    py::arg("source_features_tree") = nullptr,
    py::arg("params") = GNCParams(),
    "Estimate the transformation between two point clouds using graduated non-convexity (fast global registration).\n"
    "Features must be [N, D] arrays (e.g., FPFH features from estimate_fpfh()).");
}
