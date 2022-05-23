// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace gtsam_ext {

struct VoxelizedFrame;

/**
 * @brief Standard Frame class that holds only pointers to point attributes
 * @note  If you don't want to manage the lifetime of point data by yourself, use gtsam_ext::FrameCPU.
 */
struct Frame {
public:
  using Ptr = std::shared_ptr<Frame>;
  using ConstPtr = std::shared_ptr<const Frame>;

  Frame()
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
  virtual ~Frame() {}

  size_t size() const { return num_points; }

  // Check if the frame has attributes
  bool has_times() const;
  bool has_points() const;
  bool has_normals() const;
  bool has_covs() const;
  bool has_intensities() const;

  template<typename T>
  const T* aux_attribute(const std::string& attrib) const {
    const auto found = aux_attributes.find(attrib);
    if(found == aux_attributes.end()) {
      std::cerr << "warning: attribute " << attrib << " not found!!" << std::endl;
      return nullptr;
    }

    if (sizeof(T) != found->second.first) {
      std::cerr << "warning: attribute element size mismatch!! attrib:" << attrib << " size:" << found->second.first << " requested:" << sizeof(T) << std::endl;
    }

    return static_cast<const T*>(found->second.second);
  }

  void save(const std::string& path) const;
  void save_compact(const std::string& path) const;

public:
  size_t num_points;  // Number of points

  double* times;             // Time w.r.t. the first point (sorted)
  Eigen::Vector4d* points;   // Point coordinates (x, y, z, 1)
  Eigen::Vector4d* normals;  // Point normals (nx, ny, nz, 0)
  Eigen::Matrix4d* covs;     // Point covariances cov(3, 3) = 0
  double* intensities;       // Point intensities

  std::unordered_map<std::string, std::pair<size_t, void*>> aux_attributes;

  float* times_gpu;
  Eigen::Vector3f* points_gpu;
  Eigen::Vector3f* normals_gpu;
  Eigen::Matrix3f* covs_gpu;
  float* intensities_gpu;
};
}  // namespace gtsam_ext

#ifndef DONT_DEFINE_FRAME_TRAITS
#include <gtsam_ext/types/frame_traits.hpp>

namespace gtsam_ext {
namespace frame {

template <>
struct traits<Frame> {
  static int size(const Frame& frame) { return frame.size(); }

  static bool has_times(const Frame& frame) { return frame.has_times(); }
  static bool has_points(const Frame& frame) { return frame.has_points(); }
  static bool has_normals(const Frame& frame) { return frame.has_normals(); }
  static bool has_covs(const Frame& frame) { return frame.has_covs(); }
  static bool has_intensities(const Frame& frame) { return frame.has_intensities(); }

  static double time(const Frame& frame, size_t i) { return frame.times[i]; }
  static const Eigen::Vector4d& point(const Frame& frame, size_t i) { return frame.points[i]; }
  static const Eigen::Vector4d& normal(const Frame& frame, size_t i) { return frame.normals[i]; }
  static const Eigen::Matrix4d& cov(const Frame& frame, size_t i) { return frame.covs[i]; }
  static double intensity(const Frame& frame, size_t i) { return frame.intensities[i]; }

  static const Eigen::Vector4d* points_ptr(const Frame& frame) { return frame.points; }
};

}  // namespace frame
}  // namespace gtsam_ext
#endif
