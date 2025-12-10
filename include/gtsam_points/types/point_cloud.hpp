// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace gtsam_points {

/**
 * @brief Standard point cloud class that holds only pointers to point attributes
 * @note  If you don't want to manage the lifetime of point data by yourself, use gtsam_points::PointCloudCPU.
 */
struct PointCloud {
public:
  using Ptr = std::shared_ptr<PointCloud>;
  using ConstPtr = std::shared_ptr<const PointCloud>;

  PointCloud()
  : num_points(0),
    times(nullptr),
    points(nullptr),
    normals(nullptr),
    covs(nullptr),
    intensities(nullptr),
    times_gpu(nullptr),
    points_gpu(nullptr),
    normals_gpu(nullptr),
    covs_gpu(nullptr),
    intensities_gpu(nullptr) {}
  virtual ~PointCloud() {}

  // Forbid copy
  PointCloud(const PointCloud&) = delete;
  PointCloud& operator=(PointCloud const&) = delete;

  /// Number of points
  size_t size() const { return num_points; }

  bool has_times() const;        ///< Check if the point cloud has per-point timestamps
  bool has_points() const;       ///< Check if the point cloud has points
  bool has_normals() const;      ///< Check if the point cloud has point normals
  bool has_covs() const;         ///< Check if the point cloud has point covariances
  bool has_intensities() const;  ///< Check if the point cloud has point intensities

  bool check_times() const;        ///< Warn if the point cloud doesn't have times
  bool check_points() const;       ///< Warn if the point cloud doesn't have points
  bool check_normals() const;      ///< Warn if the point cloud doesn't have normals
  bool check_covs() const;         ///< Warn if the point cloud doesn't have covs
  bool check_intensities() const;  ///< Warn if the point cloud doesn't have intensities

  bool has_times_gpu() const;        ///< Check if the point cloud has per-point timestamps on GPU
  bool has_points_gpu() const;       ///< Check if the point cloud has points on GPU
  bool has_normals_gpu() const;      ///< Check if the point cloud has point normals on GPU
  bool has_covs_gpu() const;         ///< Check if the point cloud has point covariances on GPU
  bool has_intensities_gpu() const;  ///< Check if the point cloud has point intensities on GPU

  bool check_times_gpu() const;        ///< Warn if the point cloud doesn't have times on GPU
  bool check_points_gpu() const;       ///< Warn if the point cloud doesn't have points on GPU
  bool check_normals_gpu() const;      ///< Warn if the point cloud doesn't have normals on GPU
  bool check_covs_gpu() const;         ///< Warn if the point cloud doesn't have covs on GPU
  bool check_intensities_gpu() const;  ///< Warn if the point cloud doesn't have intensities on GPU

  /**
   * @brief Get the pointer to an aux attribute
   * @param  attrib Attribute name
   * @return Returns the pointer to it if the specified attribute exists. Otherwise, returns nullptr.
   */
  template <typename T>
  const T* aux_attribute(const std::string& attrib) const {
    const auto found = aux_attributes.find(attrib);
    if (found == aux_attributes.end()) {
      std::cerr << "warning: attribute " << attrib << " not found!!" << std::endl;
      return nullptr;
    }

    if (sizeof(T) != found->second.first) {
      std::cerr << "warning: attribute element size mismatch!! attrib:" << attrib << " size:" << found->second.first << " requested:" << sizeof(T)
                << std::endl;
    }

    return static_cast<const T*>(found->second.second);
  }

  /**
   * @brief Save the point cloud data
   * @param path Destination path
   */
  void save(const std::string& path) const;

  /**
   * @brief Save the point cloud data with a compact representation without unnecessary fields (e.g., the last element of homogeneous coordinates).
   * @param path
   */
  void save_compact(const std::string& path) const;

public:
  size_t num_points;  ///< Number of points

  double* times;             ///< Per-point timestamp w.r.t. the first point (should be sorted)
  Eigen::Vector4d* points;   ///< Point coordinates (x, y, z, 1)
  Eigen::Vector4d* normals;  ///< Point normals (nx, ny, nz, 0)
  Eigen::Matrix4d* covs;     ///< Point covariances cov(3, 3) = 0
  double* intensities;       ///< Point intensities

  /// Aux attributes <attribute_name, pair<element_size, data_ptr>>
  std::unordered_map<std::string, std::pair<size_t, void*>> aux_attributes;

  float* times_gpu;              ///< Per-point timestamp on GPU
  Eigen::Vector3f* points_gpu;   ///< Point coordinates on GPU
  Eigen::Vector3f* normals_gpu;  ///< Point normals on GPU
  Eigen::Matrix3f* covs_gpu;     ///< Point covariances on GPU
  float* intensities_gpu;        ///< Point intensities on GPU
};
}  // namespace gtsam_points

#ifndef DONT_DEFINE_FRAME_TRAITS
#include <gtsam_points/types/frame_traits.hpp>

namespace gtsam_points {
namespace frame {

template <>
struct traits<PointCloud> {
  static int size(const PointCloud& frame) { return frame.size(); }

  static bool has_times(const PointCloud& frame) { return frame.has_times(); }
  static bool has_points(const PointCloud& frame) { return frame.has_points(); }
  static bool has_normals(const PointCloud& frame) { return frame.has_normals(); }
  static bool has_covs(const PointCloud& frame) { return frame.has_covs(); }
  static bool has_intensities(const PointCloud& frame) { return frame.has_intensities(); }

  static double time(const PointCloud& frame, size_t i) { return frame.times[i]; }
  static const Eigen::Vector4d& point(const PointCloud& frame, size_t i) { return frame.points[i]; }
  static const Eigen::Vector4d& normal(const PointCloud& frame, size_t i) { return frame.normals[i]; }
  static const Eigen::Matrix4d& cov(const PointCloud& frame, size_t i) { return frame.covs[i]; }
  static double intensity(const PointCloud& frame, size_t i) { return frame.intensities[i]; }

  static const Eigen::Vector4d* points_ptr(const PointCloud& frame) { return frame.points; }
};

}  // namespace frame
}  // namespace gtsam_points
#endif
