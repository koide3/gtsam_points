// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_ext/types/frame_gpu.hpp>

#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <gtsam_ext/cuda/check_error.cuh>
#include <gtsam_ext/types/cpu_gpu_copy.hpp>

namespace gtsam_ext {

// constructor with points
template <typename T, int D>
FrameGPU::FrameGPU(const Eigen::Matrix<T, D, 1>* points, int num_points) : FrameCPU(points, num_points) {
  add_points_gpu(points, num_points);
}

template FrameGPU::FrameGPU(const Eigen::Matrix<float, 3, 1>*, int);
template FrameGPU::FrameGPU(const Eigen::Matrix<float, 4, 1>*, int);
template FrameGPU::FrameGPU(const Eigen::Matrix<double, 3, 1>*, int);
template FrameGPU::FrameGPU(const Eigen::Matrix<double, 4, 1>*, int);

// deep copy constructor
FrameGPU::FrameGPU(const Frame& frame) : FrameCPU(frame) {
  // TODO: GPU-to-GPU copy for efficiency
  if (frame.points) {
    add_points_gpu(frame.points, frame.size());
  }

  if (frame.times) {
    add_times_gpu(frame.times, frame.size());
  }

  if (frame.normals) {
    add_normals_gpu(frame.normals, frame.size());
  }

  if (frame.covs) {
    add_covs_gpu(frame.covs, frame.size());
  }

  if (frame.intensities) {
    add_intensities_gpu(frame.intensities, frame.size());
  }
}

FrameGPU::FrameGPU() {}

FrameGPU::~FrameGPU() {
  if (times_gpu) {
    check_error << cudaFreeAsync(times_gpu, 0);
  }

  if (points_gpu) {
    check_error << cudaFreeAsync(points_gpu, 0);
  }

  if (normals_gpu) {
    check_error << cudaFreeAsync(normals_gpu, 0);
  }

  if (covs_gpu) {
    check_error << cudaFreeAsync(covs_gpu, 0);
  }

  if (intensities_gpu) {
    check_error << cudaFreeAsync(intensities_gpu, 0);
  }
}

// add_times_gpu
template <typename T>
void FrameGPU::add_times_gpu(const T* times, int num_points) {
  assert(num_points == size());
  std::vector<float> times_h(num_points);
  std::copy(times, times + num_points, times_h.begin());

  if (times_gpu) {
    check_error << cudaFreeAsync(times_gpu, 0);
  }

  check_error << cudaMallocAsync(&times_gpu, sizeof(float) * num_points, 0);
  check_error << cudaMemcpyAsync(times_gpu, times_h.data(), sizeof(float) * num_points, cudaMemcpyHostToDevice, 0);
  check_error << cudaStreamSynchronize(0);
}

template void FrameGPU::add_times_gpu(const float* times, int num_points);
template void FrameGPU::add_times_gpu(const double* times, int num_points);

// add_points_gpu
template <typename T, int D>
void FrameGPU::add_points_gpu(const Eigen::Matrix<T, D, 1>* points, int num_points) {
  this->num_points = num_points;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points_h(num_points);
  for (int i = 0; i < num_points; i++) {
    points_h[i] = points[i].template cast<float>().template head<3>();
  }

  if (points_gpu) {
    check_error << cudaFreeAsync(points_gpu, 0);
  }

  check_error << cudaMallocAsync(&points_gpu, sizeof(Eigen::Vector3f) * num_points, 0);
  check_error << cudaMemcpyAsync(points_gpu, points_h.data(), sizeof(Eigen::Vector3f) * num_points, cudaMemcpyHostToDevice, 0);
  check_error << cudaStreamSynchronize(0);
}

template void FrameGPU::add_points(const Eigen::Matrix<float, 3, 1>* points, int num_points);
template void FrameGPU::add_points(const Eigen::Matrix<float, 4, 1>* points, int num_points);
template void FrameGPU::add_points(const Eigen::Matrix<double, 3, 1>* points, int num_points);
template void FrameGPU::add_points(const Eigen::Matrix<double, 4, 1>* points, int num_points);

// add_normals_gpu
template <typename T, int D>
void FrameGPU::add_normals_gpu(const Eigen::Matrix<T, D, 1>* normals, int num_points) {
  assert(num_points == this->size());

  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> normals_h(num_points);
  for (int i = 0; i < num_points; i++) {
    normals_h[i] = normals[i].template head<3>().template cast<float>();
  }

  if (normals_gpu) {
    check_error << cudaFreeAsync(normals_gpu, 0);
  }

  check_error << cudaMallocAsync(&normals_gpu, sizeof(Eigen::Vector3f) * num_points, 0);
  check_error << cudaMemcpyAsync(normals_gpu, normals_h.data(), sizeof(Eigen::Vector3f) * num_points, cudaMemcpyHostToDevice, 0);
  check_error << cudaStreamSynchronize(0);
}

template void FrameGPU::add_normals(const Eigen::Matrix<float, 3, 1>* normals, int num_points);
template void FrameGPU::add_normals(const Eigen::Matrix<float, 4, 1>* normals, int num_points);
template void FrameGPU::add_normals(const Eigen::Matrix<double, 3, 1>* normals, int num_points);
template void FrameGPU::add_normals(const Eigen::Matrix<double, 4, 1>* normals, int num_points);

// add_covs_gpu
template <typename T, int D>
void FrameGPU::add_covs_gpu(const Eigen::Matrix<T, D, D>* covs, int num_points) {
  assert(num_points == size());
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> covs_h(num_points);
  for (int i = 0; i < num_points; i++) {
    covs_h[i] = covs[i].template cast<float>().template block<3, 3>(0, 0);
  }

  if (covs_gpu) {
    check_error << cudaFreeAsync(covs_gpu, 0);
  }

  check_error << cudaMallocAsync(&covs_gpu, sizeof(Eigen::Matrix3f) * num_points, 0);
  check_error << cudaMemcpyAsync(covs_gpu, covs_h.data(), sizeof(Eigen::Matrix3f) * num_points, cudaMemcpyHostToDevice, 0);
  check_error << cudaStreamSynchronize(0);
}

template void FrameGPU::add_covs(const Eigen::Matrix<float, 3, 3>* covs, int num_points);
template void FrameGPU::add_covs(const Eigen::Matrix<float, 4, 4>* covs, int num_points);
template void FrameGPU::add_covs(const Eigen::Matrix<double, 3, 3>* covs, int num_points);
template void FrameGPU::add_covs(const Eigen::Matrix<double, 4, 4>* covs, int num_points);

// add_intensities_gpu
template <typename T>
void FrameGPU::add_intensities_gpu(const T* intensities, int num_points) {
  assert(num_points == size());
  std::vector<float> intensities_h(num_points);
  std::copy(intensities, intensities + num_points, intensities_h.begin());

  if (intensities_gpu) {
    check_error << cudaFreeAsync(intensities_gpu, 0);
  }

  check_error << cudaMallocAsync(&intensities_gpu, sizeof(float) * num_points, 0);
  check_error << cudaMemcpyAsync(intensities_gpu, intensities_h.data(), sizeof(float) * num_points, cudaMemcpyHostToDevice, 0);
  check_error << cudaStreamSynchronize(0);
}

template void FrameGPU::add_intensities(const float* intensities, int num_points);
template void FrameGPU::add_intensities(const double* intensities, int num_points);

// copy data from GPU to CPU
std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> FrameGPU::get_points_gpu() const {
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points_h;
  if (!points_gpu) {
    return points_h;
  }

  points_h.resize(num_points);
  check_error << cudaMemcpy(points_h.data(), points_gpu, sizeof(Eigen::Vector3f) * num_points, cudaMemcpyDeviceToHost);
  return points_h;
}

std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> FrameGPU::get_covs_gpu() const {
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> covs_h;
  if (!covs_gpu) {
    return covs_h;
  }

  covs_h.resize(num_points);
  check_error << cudaMemcpy(covs_h.data(), covs_gpu, sizeof(Eigen::Matrix3f) * num_points, cudaMemcpyDeviceToHost);
  return covs_h;
}

}  // namespace gtsam_ext