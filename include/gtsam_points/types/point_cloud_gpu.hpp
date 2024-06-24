// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <vector>
#include <Eigen/Core>

#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>

// forward declaration
struct CUstream_st;

namespace gtsam_points {

/**
 * @brief Point cloud frame on GPU memory
 */
struct PointCloudGPU : public PointCloudCPU {
public:
  using Ptr = std::shared_ptr<PointCloudGPU>;
  using ConstPtr = std::shared_ptr<const PointCloudGPU>;

  PointCloudGPU();
  ~PointCloudGPU();

  template <typename T, int D>
  PointCloudGPU(const Eigen::Matrix<T, D, 1>* points, int num_points);

  template <typename T, int D, typename Alloc>
  PointCloudGPU(const std::vector<Eigen::Matrix<T, D, 1>, Alloc>& points) : PointCloudGPU(points.data(), points.size()) {}

  // Forbid shallow copy
  PointCloudGPU(const PointCloudGPU& points) = delete;
  PointCloudGPU& operator=(PointCloudGPU const&) = delete;

  // Deep copy
  static PointCloudGPU::Ptr clone(const PointCloud& frame, CUstream_st* stream = 0);

  // add_*_gpu() only adds attributes to GPU storage
  template <typename T>
  void add_times(const T* times, int num_points, CUstream_st* stream = 0) {
    PointCloudCPU::add_times<T>(times, num_points);
    add_times_gpu(times, num_points, stream);
  }
  template <typename T>
  void add_times(const std::vector<T>& times, CUstream_st* stream = 0) {
    add_times(times.data(), times.size(), stream);
  }

  template <typename T>
  void add_times_gpu(const T* times, int num_points, CUstream_st* stream = 0);
  template <typename T>
  void add_times_gpu(const std::vector<T>& times, CUstream_st* stream = 0) {
    add_times_gpu(times.data(), times.size(), stream);
  }

  template <typename T, int D>
  void add_points(const Eigen::Matrix<T, D, 1>* points, int num_points, CUstream_st* stream = 0) {
    PointCloudCPU::add_points<T, D>(points, num_points);
    add_points_gpu(points, num_points, stream);
  }
  template <typename T, int D, typename Alloc>
  void add_points(const std::vector<Eigen::Matrix<T, D, 1>, Alloc>& points, CUstream_st* stream = 0) {
    add_points(points.data(), points.size(), stream);
  }

  template <typename T, int D>
  void add_points_gpu(const Eigen::Matrix<T, D, 1>* points, int num_points, CUstream_st* stream = 0);
  template <typename T, int D, typename Alloc>
  void add_points_gpu(const std::vector<Eigen::Matrix<T, D, 1>, Alloc>& points, CUstream_st* stream = 0) {
    add_points_gpu(points.data(), points.size(), stream);
  }

  template <typename T, int D>
  void add_normals(const Eigen::Matrix<T, D, 1>* normals, int num_points, CUstream_st* stream = 0) {
    PointCloudCPU::add_normals<T, D>(normals, num_points);
    add_normals_gpu<T, D>(normals, num_points, stream);
  }
  template <typename T, int D, typename Alloc>
  void add_normals(const std::vector<Eigen::Matrix<T, D, 1>, Alloc>& normals, CUstream_st* stream = 0) {
    add_normals(normals.data(), normals.size(), stream);
  }

  template <typename T, int D>
  void add_normals_gpu(const Eigen::Matrix<T, D, 1>* normals, int num_points, CUstream_st* stream = 0);
  template <typename T, int D, typename Alloc>
  void add_normals_gpu(const std::vector<Eigen::Matrix<T, D, 1>, Alloc>& normals, CUstream_st* stream = 0) {
    add_normals_gpu(normals.data(), normals.size(), stream);
  }

  template <typename T, int D>
  void add_covs(const Eigen::Matrix<T, D, D>* covs, int num_points, CUstream_st* stream = 0) {
    PointCloudCPU::add_covs<T, D>(covs, num_points);
    add_covs_gpu<T, D>(covs, num_points, stream);
  }
  template <typename T, int D, typename Alloc>
  void add_covs(const std::vector<Eigen::Matrix<T, D, D>, Alloc>& covs, CUstream_st* stream = 0) {
    add_covs(covs.data(), covs.size(), stream);
  }

  template <typename T, int D>
  void add_covs_gpu(const Eigen::Matrix<T, D, D>* covs, int num_points, CUstream_st* stream = 0);
  template <typename T, int D, typename Alloc>
  void add_covs_gpu(const std::vector<Eigen::Matrix<T, D, D>, Alloc>& covs, CUstream_st* stream = 0) {
    add_covs_gpu(covs.data(), covs.size(), stream);
  }

  template <typename T>
  void add_intensities(const T* intensities, int num_points, CUstream_st* stream = 0) {
    PointCloudCPU::add_intensities<T>(intensities, num_points);
    add_intensities_gpu(intensities, num_points, stream);
  }
  template <typename T>
  void add_intensities(const std::vector<T>& intensities, CUstream_st* stream = 0) {
    add_intensities(intensities.data(), intensities.size(), stream);
  }

  template <typename T>
  void add_intensities_gpu(const T* intensities, int num_points, CUstream_st* stream = 0);
  template <typename T>
  void add_intensities_gpu(const std::vector<T>& intensities, CUstream_st* stream = 0) {
    add_intensities_gpu(intensities.data(), intensities.size(), stream);
  }

  void download_points(CUstream_st* stream = 0);
};

// Device to host data transfer
std::vector<Eigen::Vector3f> download_points_gpu(const gtsam_points::PointCloud& frame, CUstream_st* stream = nullptr);
std::vector<Eigen::Matrix3f> download_covs_gpu(const gtsam_points::PointCloud& frame, CUstream_st* stream = nullptr);
std::vector<Eigen::Vector3f> download_normals_gpu(const gtsam_points::PointCloud& frame, CUstream_st* stream = nullptr);
std::vector<float> download_intensities_gpu(const gtsam_points::PointCloud& frame, CUstream_st* stream = nullptr);
std::vector<float> download_times_gpu(const gtsam_points::PointCloud& frame, CUstream_st* stream = nullptr);

}  // namespace gtsam_points
