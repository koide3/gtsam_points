// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <vector>
#include <Eigen/Core>

#include <gtsam_ext/types/frame.hpp>

// forward declaration
namespace thrust {

template <typename T>
class device_allocator;

template <typename T, typename Alloc>
class device_vector;

}  // namespace thrust

namespace gtsam_ext {

struct FrameGPU : public Frame {
public:
  using Ptr = std::shared_ptr<FrameGPU>;
  using ConstPtr = std::shared_ptr<const FrameGPU>;

  template <typename T, int D>
  FrameGPU(const Eigen::Matrix<T, D, 1>* points, int num_points, bool allocate_cpu = true);

  template <typename T, int D, typename Alloc>
  FrameGPU(const std::vector<Eigen::Matrix<T, D, 1>, Alloc>& points, bool allocate_cpu = true) : FrameGPU(points.data(), points.size(), allocate_cpu) {}

  FrameGPU(const Frame& frame, bool allocate_cpu = true);

  FrameGPU();
  ~FrameGPU();

  // add_*_gpu() only adds attributes to GPU storage
  template <typename T>
  void add_times(const T* times, int num_points);
  template <typename T>
  void add_times_gpu(const T* times, int num_points);
  template <typename T>
  void add_times(const std::vector<T>& times) {
    add_times(times.data(), times.size());
  }
  template <typename T>
  void add_times_gpu(const std::vector<T>& times) {
    add_times_gpu(times.data(), times.size());
  }

  template <typename T, int D>
  void add_points(const Eigen::Matrix<T, D, 1>* points, int num_points);
  template <typename T, int D>
  void add_points_gpu(const Eigen::Matrix<T, D, 1>* points, int num_points);
  template <typename T, int D, typename Alloc>
  void add_points(const std::vector<Eigen::Matrix<T, D, 1>, Alloc>& points) {
    add_points(points.data(), points.size());
  }
  template <typename T, int D, typename Alloc>
  void add_points_gpu(const std::vector<Eigen::Matrix<T, D, 1>, Alloc>& points) {
    add_points_gpu(points.data(), points.size());
  }

  template <typename T, int D>
  void add_normals(const Eigen::Matrix<T, D, 1>* normals, int num_points);
  template <typename T, int D>
  void add_normals_gpu(const Eigen::Matrix<T, D, 1>* normals, int num_points);
  template <typename T, int D, typename Alloc>
  void add_normals(const std::vector<Eigen::Matrix<T, D, 1>, Alloc>& normals) {
    add_normals(normals.data(), normals.size());
  }
  template <typename T, int D, typename Alloc>
  void add_normals_gpu(const std::vector<Eigen::Matrix<T, D, 1>, Alloc>& normals) {
    add_normals_gpu(normals.data(), normals.size());
  }

  template <typename T, int D>
  void add_covs(const Eigen::Matrix<T, D, D>* covs, int num_points);
  template <typename T, int D>
  void add_covs_gpu(const Eigen::Matrix<T, D, D>* covs, int num_points);
  template <typename T, int D, typename Alloc>
  void add_covs(const std::vector<Eigen::Matrix<T, D, D>, Alloc>& covs) {
    add_covs(covs.data(), covs.size());
  }
  template <typename T, int D, typename Alloc>
  void add_covs_gpu(const std::vector<Eigen::Matrix<T, D, D>, Alloc>& covs) {
    add_covs_gpu(covs.data(), covs.size());
  }

  template <typename T>
  void add_intensities(const T* intensities, int num_points);
  template <typename T>
  void add_intensities_gpu(const T* intensities, int num_points);
  template <typename T>
  void add_intensities(const std::vector<T>& intensities) {
    return add_intensities(intensities.data(), intensities.size());
  }
  template <typename T>
  void add_intensities_gpu(const std::vector<T>& intensities) {
    return add_intensities_gpu(intensities.data(), intensities.size());
  }

  // copy data from GPU to CPU
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> get_points_gpu() const;
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> get_covs_gpu() const;

public:
  using FloatsGPU = thrust::device_vector<float, thrust::device_allocator<float>>;
  using PointsGPU = thrust::device_vector<Eigen::Vector3f, thrust::device_allocator<Eigen::Vector3f>>;
  using MatricesGPU = thrust::device_vector<Eigen::Matrix3f, thrust::device_allocator<Eigen::Matrix3f>>;

  std::vector<double> times_storage;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points_storage;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> normals_storage;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> covs_storage;
  std::vector<double> intensities_storage;

  std::unique_ptr<FloatsGPU> times_gpu_storage;
  std::unique_ptr<PointsGPU> points_gpu_storage;
  std::unique_ptr<PointsGPU> normals_gpu_storage;
  std::unique_ptr<MatricesGPU> covs_gpu_storage;
  std::unique_ptr<FloatsGPU> intensities_gpu_storage;
};

}  // namespace gtsam_ext
