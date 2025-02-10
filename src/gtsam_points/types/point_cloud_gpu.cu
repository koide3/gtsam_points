// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/types/point_cloud_gpu.hpp>

#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <gtsam_points/cuda/check_error.cuh>
#include <gtsam_points/cuda/cuda_malloc_async.hpp>

namespace gtsam_points {

// constructor with points
template <typename T, int D>
PointCloudGPU::PointCloudGPU(const Eigen::Matrix<T, D, 1>* points, int num_points) : PointCloudCPU(points, num_points) {
  add_points_gpu(points, num_points);
}

template PointCloudGPU::PointCloudGPU(const Eigen::Matrix<float, 3, 1>*, int);
template PointCloudGPU::PointCloudGPU(const Eigen::Matrix<float, 4, 1>*, int);
template PointCloudGPU::PointCloudGPU(const Eigen::Matrix<double, 3, 1>*, int);
template PointCloudGPU::PointCloudGPU(const Eigen::Matrix<double, 4, 1>*, int);

// deep copy
PointCloudGPU::Ptr PointCloudGPU::clone(const PointCloud& frame, CUstream_st* stream) {
  auto new_frame = std::make_shared<gtsam_points::PointCloudGPU>();

  // TODO: GPU-to-GPU copy for efficiency
  if (frame.points) {
    new_frame->add_points(frame.points, frame.size(), stream);
  }

  if (frame.times) {
    new_frame->add_times(frame.times, frame.size(), stream);
  }

  if (frame.normals) {
    new_frame->add_normals(frame.normals, frame.size(), stream);
  }

  if (frame.covs) {
    new_frame->add_covs(frame.covs, frame.size(), stream);
  }

  if (frame.intensities) {
    new_frame->add_intensities(frame.intensities, frame.size(), stream);
  }

  for (const auto& aux : frame.aux_attributes) {
    const auto& name = aux.first;
    const auto& size_data = aux.second;

    auto buffer = std::make_shared<std::vector<unsigned char, Eigen::aligned_allocator<unsigned char>>>(size_data.first * frame.size());
    memcpy(buffer->data(), size_data.second, size_data.first * frame.size());

    new_frame->aux_attributes_storage[name] = buffer;
    new_frame->aux_attributes[name] = {size_data.first, buffer->data()};
  }

  return new_frame;
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
    std::vector<Eigen::Vector3f> points_h(num_points);
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
    std::vector<Eigen::Vector3f> normals_h(num_points);
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
    std::vector<Eigen::Matrix3f> covs_h(num_points);
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

std::vector<Eigen::Vector3f> download_points_gpu(const gtsam_points::PointCloud& frame, CUstream_st* stream) {
  if (!frame.points_gpu) {
    std::cerr << "error: frame does not have points on GPU!!" << std::endl;
    return {};
  }

  std::vector<Eigen::Vector3f> points(frame.size());
  check_error << cudaMemcpyAsync(points.data(), frame.points_gpu, sizeof(Eigen::Vector3f) * frame.size(), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return points;
}

std::vector<Eigen::Matrix3f> download_covs_gpu(const gtsam_points::PointCloud& frame, CUstream_st* stream) {
  if (!frame.covs_gpu) {
    std::cerr << "error: frame does not have covs on GPU!!" << std::endl;
    return {};
  }

  std::vector<Eigen::Matrix3f> covs(frame.size());
  check_error << cudaMemcpyAsync(covs.data(), frame.covs_gpu, sizeof(Eigen::Matrix3f) * frame.size(), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return covs;
}

std::vector<Eigen::Vector3f> download_normals_gpu(const gtsam_points::PointCloud& frame, CUstream_st* stream) {
  if (!frame.normals_gpu) {
    std::cerr << "error: frame does not have normals on GPU!!" << std::endl;
    return {};
  }

  std::vector<Eigen::Vector3f> normals(frame.size());
  check_error << cudaMemcpyAsync(normals.data(), frame.normals_gpu, sizeof(Eigen::Vector3f) * frame.size(), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return normals;
}

std::vector<float> download_intensities_gpu(const gtsam_points::PointCloud& frame, CUstream_st* stream) {
  if (!frame.intensities_gpu) {
    std::cerr << "error: frame does not have intensities on GPU!!" << std::endl;
    return {};
  }

  std::vector<float> intensities(frame.size());
  check_error << cudaMemcpyAsync(intensities.data(), frame.intensities_gpu, sizeof(float) * frame.size(), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return intensities;
}

std::vector<float> download_times_gpu(const gtsam_points::PointCloud& frame, CUstream_st* stream) {
  if (!frame.times_gpu) {
    std::cerr << "error: frame does not have times on GPU!!" << std::endl;
    return {};
  }

  std::vector<float> times(frame.size());
  check_error << cudaMemcpyAsync(times.data(), frame.times_gpu, sizeof(float) * frame.size(), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return times;
}

}  // namespace gtsam_points