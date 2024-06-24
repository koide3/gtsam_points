// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

/**
 * @file  basic_frame_manipulation.cpp
 * @brief This example demonstrates how to create and manipulate gtsam_points::PointCloud class to manage point clouds.
 */

#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/factors/integrated_icp_factor.hpp>

int main(int argc, char** argv) {
  const int num_points = 128;

  // gtsam_points::PointCloudCPU can be created from a vector of Eigen::Vector3f, 4f, 3d or 4d.
  // Input points are converted and held as Eigen::Vector4d internally.
  // Note that if you feed 4D vectors, their last elements (w) must be 1.
  std::vector<Eigen::Vector3f> points_3f(num_points);
  const auto frame_3f = std::make_shared<gtsam_points::PointCloudCPU>(points_3f);

  std::vector<Eigen::Vector4f> points_4f(num_points);
  const auto frame_4f = std::make_shared<gtsam_points::PointCloudCPU>(points_4f);

  std::vector<Eigen::Vector3d> points_3d(num_points);
  const auto frame_3d = std::make_shared<gtsam_points::PointCloudCPU>(points_3d);

  std::vector<Eigen::Vector4d> points_4d(num_points);
  const auto frame_4d = std::make_shared<gtsam_points::PointCloudCPU>(points_4d);

  // Is it also possible to create FrameCPU from a raw Eigen::Vector pointer.
  const auto frame = std::make_shared<gtsam_points::PointCloudCPU>(points_4d.data(), points_4d.size());

  // You can add point attributes by calling FrameCPU::add_*.
  std::vector<double> times(num_points);
  std::vector<Eigen::Vector4d> normals(num_points);
  std::vector<Eigen::Matrix4d> covs(num_points);
  std::vector<double> intensities(num_points);
  frame->add_times(times);              // Add point times
  frame->add_normals(normals);          // Add point normals
  frame->add_covs(covs);                // Add point covariances
  frame->add_intensities(intensities);  // Add point intensities

  // gtsam_points::PointCloudCPU is derived from gtsam_points::PointCloud that holds only pointers to point attributes.
  gtsam_points::PointCloud::Ptr base_frame = frame;
  int frame_size = base_frame->num_points;             // Number of points
  frame_size = base_frame->size();                     // frame->size() == frame->num_points
  Eigen::Vector4d* points_ptr = base_frame->points;    // Pointer to point coordinates
  double* times_ptr = base_frame->times;               // Pointer to point times
  Eigen::Vector4d* normals_ptr = base_frame->normals;  // Pointer to point normals
  Eigen::Matrix4d* covs_ptr = base_frame->covs;        // Pointer to point covariances
  double* intensities_ptr = base_frame->intensities;   // Pointer to point intensities

  // If you don't want to let gtsam_points::PointCloudCPU hold point data but want to manage points by yourself,
  // you can just give gtsam_points::PointCloud pointers to point attributes.
  // Note that you need to take care of the ownership of point data in this case.
  gtsam_points::PointCloud::Ptr base_frame2(new gtsam_points::PointCloud);
  base_frame2->num_points = points_4d.size();
  base_frame2->points = points_4d.data();

  auto icp_factor = gtsam::make_shared<gtsam_points::IntegratedICPFactor>(0, 1, base_frame, base_frame2);

  // If you are interested in using your custom classes with gtsam_points, see "advanced_frame_manipulation.cpp".

  return 0;
}