// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_ext/types/point_cloud_gpu.hpp>

#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <gtsam_ext/cuda/check_error.cuh>
#include <gtsam_ext/cuda/cuda_malloc_async.hpp>

namespace gtsam_ext {

// constructor with points
template <typename T, int D>
PointCloudGPU::PointCloudGPU(const Eigen::Matrix<T, D, 1>* points, int num_points) : PointCloudCPU(points, num_points) {
  add_points_gpu(points, num_points);
}

template PointCloudGPU::PointCloudGPU(const Eigen::Matrix<float, 3, 1>*, int);
template PointCloudGPU::PointCloudGPU(const Eigen::Matrix<float, 4, 1>*, int);
template PointCloudGPU::PointCloudGPU(const Eigen::Matrix<double, 3, 1>*, int);
template PointCloudGPU::PointCloudGPU(const Eigen::Matrix<double, 4, 1>*, int);

// deep copy constructor
PointCloudGPU::PointCloudGPU(const PointCloud& frame, CUstream_st* stream) : PointCloudCPU(frame) {
  // TODO: GPU-to-GPU copy for efficiency
  if (frame.points) {
    add_points_gpu(frame.points, frame.size(), stream);
  }

  if (frame.times) {
    add_times_gpu(frame.times, frame.size(), stream);
  }

  if (frame.normals) {
    add_normals_gpu(frame.normals, frame.size(), stream);
  }

  if (frame.covs) {
    add_covs_gpu(frame.covs, frame.size(), stream);
  }

  if (frame.intensities) {
    add_intensities_gpu(frame.intensities, frame.size(), stream);
  }
}

PointCloudGPU::PointCloudGPU() {}

PointCloudGPU::~PointCloudGPU() {
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
void PointCloudGPU::add_times_gpu(const T* times, int num_points, CUstream_st* stream) {
  assert(num_points == size());
  if (times_gpu) {
    check_error << cudaFreeAsync(times_gpu, stream);
  }

  check_error << cudaMallocAsync(&times_gpu, sizeof(float) * num_points, stream);
  if (times) {
    std::vector<float> times_h(num_points);
    std::copy(times, times + num_points, times_h.begin());
    check_error << cudaMemcpyAsync(times_gpu, times_h.data(), sizeof(float) * num_points, cudaMemcpyHostToDevice, stream);
    check_error << cudaStreamSynchronize(stream);
  }
}

template void PointCloudGPU::add_times_gpu(const float* times, int num_points, CUstream_st* stream);
template void PointCloudGPU::add_times_gpu(const double* times, int num_points, CUstream_st* stream);

// add_points_gpu
template <typename T, int D>
void PointCloudGPU::add_points_gpu(const Eigen::Matrix<T, D, 1>* points, int num_points, CUstream_st* stream) {
  this->num_points = num_points;
  if (points_gpu) {
    check_error << cudaFreeAsync(points_gpu, stream);
  }

  check_error << cudaMallocAsync(&points_gpu, sizeof(Eigen::Vector3f) * num_points, stream);
  if (points) {
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points_h(num_points);
    for (int i = 0; i < num_points; i++) {
      points_h[i] = points[i].template cast<float>().template head<3>();
    }
    check_error << cudaMemcpyAsync(points_gpu, points_h.data(), sizeof(Eigen::Vector3f) * num_points, cudaMemcpyHostToDevice, stream);
    check_error << cudaStreamSynchronize(stream);
  }
}

template void PointCloudGPU::add_points_gpu(const Eigen::Matrix<float, 3, 1>* points, int num_points, CUstream_st* stream);
template void PointCloudGPU::add_points_gpu(const Eigen::Matrix<float, 4, 1>* points, int num_points, CUstream_st* stream);
template void PointCloudGPU::add_points_gpu(const Eigen::Matrix<double, 3, 1>* points, int num_points, CUstream_st* stream);
template void PointCloudGPU::add_points_gpu(const Eigen::Matrix<double, 4, 1>* points, int num_points, CUstream_st* stream);

// add_normals_gpu
template <typename T, int D>
void PointCloudGPU::add_normals_gpu(const Eigen::Matrix<T, D, 1>* normals, int num_points, CUstream_st* stream) {
  assert(num_points == this->size());

  if (normals_gpu) {
    check_error << cudaFreeAsync(normals_gpu, stream);
  }

  check_error << cudaMallocAsync(&normals_gpu, sizeof(Eigen::Vector3f) * num_points, stream);
  if (normals) {
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> normals_h(num_points);
    for (int i = 0; i < num_points; i++) {
      normals_h[i] = normals[i].template head<3>().template cast<float>();
    }
    check_error << cudaMemcpyAsync(normals_gpu, normals_h.data(), sizeof(Eigen::Vector3f) * num_points, cudaMemcpyHostToDevice, stream);
    check_error << cudaStreamSynchronize(stream);
  }
}

template void PointCloudGPU::add_normals_gpu(const Eigen::Matrix<float, 3, 1>* normals, int num_points, CUstream_st* stream);
template void PointCloudGPU::add_normals_gpu(const Eigen::Matrix<float, 4, 1>* normals, int num_points, CUstream_st* stream);
template void PointCloudGPU::add_normals_gpu(const Eigen::Matrix<double, 3, 1>* normals, int num_points, CUstream_st* stream);
template void PointCloudGPU::add_normals_gpu(const Eigen::Matrix<double, 4, 1>* normals, int num_points, CUstream_st* stream);

// add_covs_gpu
template <typename T, int D>
void PointCloudGPU::add_covs_gpu(const Eigen::Matrix<T, D, D>* covs, int num_points, CUstream_st* stream) {
  assert(num_points == size());

  if (covs_gpu) {
    check_error << cudaFreeAsync(covs_gpu, stream);
  }

  check_error << cudaMallocAsync(&covs_gpu, sizeof(Eigen::Matrix3f) * num_points, stream);
  if (covs) {
    std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> covs_h(num_points);
    for (int i = 0; i < num_points; i++) {
      covs_h[i] = covs[i].template cast<float>().template block<3, 3>(0, 0);
    }
    check_error << cudaMemcpyAsync(covs_gpu, covs_h.data(), sizeof(Eigen::Matrix3f) * num_points, cudaMemcpyHostToDevice, stream);
    check_error << cudaStreamSynchronize(stream);
  }
}

template void PointCloudGPU::add_covs_gpu(const Eigen::Matrix<float, 3, 3>* covs, int num_points, CUstream_st* stream);
template void PointCloudGPU::add_covs_gpu(const Eigen::Matrix<float, 4, 4>* covs, int num_points, CUstream_st* stream);
template void PointCloudGPU::add_covs_gpu(const Eigen::Matrix<double, 3, 3>* covs, int num_points, CUstream_st* stream);
template void PointCloudGPU::add_covs_gpu(const Eigen::Matrix<double, 4, 4>* covs, int num_points, CUstream_st* stream);

// add_intensities_gpu
template <typename T>
void PointCloudGPU::add_intensities_gpu(const T* intensities, int num_points, CUstream_st* stream) {
  assert(num_points == size());

  if (intensities_gpu) {
    check_error << cudaFreeAsync(intensities_gpu, stream);
  }

  check_error << cudaMallocAsync(&intensities_gpu, sizeof(float) * num_points, stream);
  if (intensities) {
    std::vector<float> intensities_h(num_points);
    std::copy(intensities, intensities + num_points, intensities_h.begin());
    check_error << cudaMemcpyAsync(intensities_gpu, intensities_h.data(), sizeof(float) * num_points, cudaMemcpyHostToDevice, stream);
    check_error << cudaStreamSynchronize(stream);
  }
}

template void PointCloudGPU::add_intensities_gpu(const float* intensities, int num_points, CUstream_st* stream);
template void PointCloudGPU::add_intensities_gpu(const double* intensities, int num_points, CUstream_st* stream);

void PointCloudGPU::download_points(CUstream_st* stream) {
  if (!points_gpu) {
    std::cerr << "error: frame does not have points on GPU!!" << std::endl;
    return;
  }

  std::vector<Eigen::Vector3f> points_h(num_points);
  check_error << cudaMemcpyAsync(points_h.data(), points_gpu, sizeof(Eigen::Vector3f) * num_points, cudaMemcpyDeviceToHost, stream);

  if (!points) {
    points_storage.resize(num_points);
    points = points_storage.data();
  }

  std::transform(points_h.begin(), points_h.end(), points, [](const Eigen::Vector3f& p) { return Eigen::Vector4d(p.x(), p.y(), p.z(), 1.0); });
}

std::vector<Eigen::Vector3f> download_points_gpu(const gtsam_ext::PointCloud& frame, CUstream_st* stream) {
  if (!frame.points_gpu) {
    std::cerr << "error: frame does not have points on GPU!!" << std::endl;
    return {};
  }

  std::vector<Eigen::Vector3f> points(frame.size());
  check_error << cudaMemcpyAsync(points.data(), frame.points_gpu, sizeof(Eigen::Vector3f) * frame.size(), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return points;
}

std::vector<Eigen::Matrix3f> download_covs_gpu(const gtsam_ext::PointCloud& frame, CUstream_st* stream) {
  if (!frame.covs_gpu) {
    std::cerr << "error: frame does not have covs on GPU!!" << std::endl;
    return {};
  }

  std::vector<Eigen::Matrix3f> covs(frame.size());
  check_error << cudaMemcpyAsync(covs.data(), frame.covs_gpu, sizeof(Eigen::Matrix3f) * frame.size(), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return covs;
}

std::vector<Eigen::Vector3f> download_normals_gpu(const gtsam_ext::PointCloud& frame, CUstream_st* stream) {
  if (!frame.normals_gpu) {
    std::cerr << "error: frame does not have normals on GPU!!" << std::endl;
    return {};
  }

  std::vector<Eigen::Vector3f> normals(frame.size());
  check_error << cudaMemcpyAsync(normals.data(), frame.normals_gpu, sizeof(Eigen::Vector3f) * frame.size(), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return normals;
}

}  // namespace gtsam_ext