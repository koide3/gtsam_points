// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_ext/types/frame_gpu.hpp>

#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <gtsam_ext/types/cpu_gpu_copy.hpp>

namespace gtsam_ext {

template <typename T, int D>
FrameGPU::FrameGPU(const Eigen::Matrix<T, D, 1>* points, int num_points, bool allocate_cpu)
: times_gpu_storage(new FloatsGPU()),
  points_gpu_storage(new PointsGPU()),
  normals_gpu_storage(new PointsGPU()),
  covs_gpu_storage(new MatricesGPU()) {
  //
  if (allocate_cpu) {
    add_points(points, num_points);
  } else {
    add_points_gpu(points, num_points);
  }
}

template <typename T, int D>
FrameGPU::FrameGPU(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& points, bool allocate_cpu)
: FrameGPU(points.data(), points.size(), allocate_cpu) {}

FrameGPU::FrameGPU(const Frame& frame, bool allocate_cpu)
: times_gpu_storage(new FloatsGPU()),
  points_gpu_storage(new PointsGPU()),
  normals_gpu_storage(new PointsGPU()),
  covs_gpu_storage(new MatricesGPU()) {
  //
  num_points = frame.size();

  if (allocate_cpu) {
    if (frame.points) {
      add_points(frame.points, frame.size());
    }

    if (frame.times) {
      add_times(frame.times, frame.size());
    }

    if (frame.normals) {
      add_normals(frame.normals, frame.size());
    }

    if (frame.covs) {
      add_covs(frame.covs, frame.size());
    }
  } else {
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
  }

  /*
  copy_to_gpu(*times_gpu_storage, &times_gpu, frame.times, frame.times_gpu, num_points);
  copy_to_gpu(*points_gpu_storage, &points_gpu, frame.points, frame.points_gpu, num_points);
  copy_to_gpu(*normals_gpu_storage, &normals_gpu, frame.normals, frame.normals_gpu, num_points);
  copy_to_gpu(*covs_gpu_storage, &covs_gpu, frame.covs, frame.covs_gpu, num_points);

  if (allocate_cpu) {
    copy_to_cpu(times_storage, &times, frame.times, frame.times_gpu, num_points, 0.0);
    copy_to_cpu(points_storage, &points, frame.points, frame.points_gpu, num_points, Eigen::Vector4d(0.0, 0.0, 0.0, 1.0));
    copy_to_cpu(normals_storage, &normals, frame.normals, frame.normals_gpu, num_points, Eigen::Vector4d::Zero().eval());
    copy_to_cpu(covs_storage, &covs, frame.covs, frame.covs_gpu, num_points, Eigen::Matrix4d::Zero().eval());
  }
  */
}

FrameGPU::FrameGPU() : times_gpu_storage(new FloatsGPU()), points_gpu_storage(new PointsGPU()), normals_gpu_storage(new PointsGPU()), covs_gpu_storage(new MatricesGPU()) {}

FrameGPU::~FrameGPU() {}

template <typename T>
void FrameGPU::add_times(const T* times, int num_points) {
  add_times_gpu(times, num_points);

  times_storage.resize(num_points);
  thrust::copy(times, times + num_points, times_storage.begin());
  this->times = times_storage.data();
}

template <typename T>
void FrameGPU::add_times_gpu(const T* times, int num_points) {
  assert(num_points == size());
  thrust::host_vector<float> times_h(num_points);
  std::copy(times, times + num_points, times_h.begin());

  times_gpu_storage->resize(num_points);
  cudaMemcpy(thrust::raw_pointer_cast(times_gpu_storage->data()), times_h.data(), sizeof(float) * num_points, cudaMemcpyHostToDevice);
  this->times_gpu = thrust::raw_pointer_cast(times_gpu_storage->data());
}

template <typename T>
void FrameGPU::add_times(const std::vector<T>& times) {
  add_times(times.data(), times.size());
}

template <typename T>
void FrameGPU::add_times_gpu(const std::vector<T>& times) {
  add_times_gpu(times.data(), times.size());
}

template <typename T, int D>
void FrameGPU::add_points(const Eigen::Matrix<T, D, 1>* points, int num_points) {
  add_points_gpu(points, num_points);

  points_storage.resize(num_points, Eigen::Vector4d::UnitW());
  this->points = points_storage.data();
  for (int i = 0; i < num_points; i++) {
    points_storage[i].template head<D>() = points[i].template cast<double>();
  }
}

template <typename T, int D>
void FrameGPU::add_points_gpu(const Eigen::Matrix<T, D, 1>* points, int num_points) {
  this->num_points = num_points;
  points_gpu_storage->resize(num_points);
  points_gpu = thrust::raw_pointer_cast(points_gpu_storage->data());

  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points_f(num_points);
  for (int i = 0; i < num_points; i++) {
    points_f[i] = points[i].template head<3>().template cast<float>();
  }
  cudaMemcpy(points_gpu, points_f.data(), sizeof(Eigen::Vector3f) * num_points, cudaMemcpyHostToDevice);
}

template <typename T, int D>
void FrameGPU::add_points(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& points) {
  add_points(points.data(), points.size());
}

template <typename T, int D>
void FrameGPU::add_points_gpu(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& points) {
  add_points_gpu(points.data(), points.size());
}

template <typename T, int D>
void FrameGPU::add_normals(const Eigen::Matrix<T, D, 1>* normals, int num_points) {
  add_normals_gpu(normals, num_points);

  normals_storage.resize(num_points, Eigen::Vector4d(0, 0, 0, 0));
  for (int i = 0; i < num_points; i++) {
    normals_storage[i].template head<D>() = normals[i].template cast<double>();
  }
  this->normals = normals_storage.data();
}

template <typename T, int D>
void FrameGPU::add_normals_gpu(const Eigen::Matrix<T, D, 1>* normals, int num_points) {
  assert(normals.size() == size());

  thrust::host_vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> normals_h(num_points);
  for (int i = 0; i < num_points; i++) {
    normals_h[i] = normals[i].template head<3>().template cast<float>();
  }

  normals_gpu_storage->resize(num_points);
  cudaMemcpy(thrust::raw_pointer_cast(normals_gpu_storage->data()), normals_h.data(), sizeof(Eigen::Vector3f) * num_points, cudaMemcpyHostToDevice);
  this->normals_gpu = thrust::raw_pointer_cast(normals_gpu_storage->data());
}

template <typename T, int D>
void FrameGPU::add_normals(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& normals) {
  add_normals(normals.data(), normals.size());
}

template <typename T, int D>
void FrameGPU::add_normals_gpu(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& normals) {
  add_normals_gpu(normals.data(), normals.size());
}

template <typename T, int D>
void FrameGPU::add_covs(const Eigen::Matrix<T, D, D>* covs, int num_points) {
  add_covs_gpu(covs, num_points);

  covs_storage.resize(num_points, Eigen::Matrix4d::Zero());
  for (int i = 0; i < num_points; i++) {
    covs_storage[i].template block<D, D>(0, 0) = covs[i].template cast<double>();
  }
  this->covs = covs_storage.data();
}

template <typename T, int D>
void FrameGPU::add_covs_gpu(const Eigen::Matrix<T, D, D>* covs, int num_points) {
  assert(covs.size() == size());
  thrust::host_vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> covs_h(num_points);
  for (int i = 0; i < num_points; i++) {
    covs_h[i] = covs[i].template block<3, 3>(0, 0).template cast<float>();
  }

  covs_gpu_storage->resize(num_points);
  cudaMemcpy(thrust::raw_pointer_cast(covs_gpu_storage->data()), covs_h.data(), sizeof(Eigen::Matrix3f) * num_points, cudaMemcpyHostToDevice);
  this->covs_gpu = thrust::raw_pointer_cast(covs_gpu_storage->data());
}

template <typename T, int D>
void FrameGPU::add_covs(const std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>>& covs) {
  add_covs(covs.data(), covs.size());
}

template <typename T, int D>
void FrameGPU::add_covs_gpu(const std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>>& covs) {
  add_covs_gpu(covs.data(), covs.size());
}

template FrameGPU::FrameGPU(const Eigen::Matrix<float, 3, 1>*, int, bool);
template FrameGPU::FrameGPU(const Eigen::Matrix<float, 4, 1>*, int, bool);
template FrameGPU::FrameGPU(const Eigen::Matrix<double, 3, 1>*, int, bool);
template FrameGPU::FrameGPU(const Eigen::Matrix<double, 4, 1>*, int, bool);

template FrameGPU::FrameGPU(const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>&, bool);
template FrameGPU::FrameGPU(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>&, bool);
template FrameGPU::FrameGPU(const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>&, bool);
template FrameGPU::FrameGPU(const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>&, bool);

template void FrameGPU::add_times(const std::vector<float>&);
template void FrameGPU::add_times(const std::vector<double>&);
template void FrameGPU::add_times_gpu(const std::vector<float>&);
template void FrameGPU::add_times_gpu(const std::vector<double>&);

template void FrameGPU::add_points(const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>&);
template void FrameGPU::add_points(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>&);
template void FrameGPU::add_points(const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>&);
template void FrameGPU::add_points(const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>&);
template void FrameGPU::add_points_gpu(const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>&);
template void FrameGPU::add_points_gpu(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>&);
template void FrameGPU::add_points_gpu(const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>&);
template void FrameGPU::add_points_gpu(const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>&);

template void FrameGPU::add_normals(const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>&);
template void FrameGPU::add_normals(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>&);
template void FrameGPU::add_normals(const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>&);
template void FrameGPU::add_normals(const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>&);
template void FrameGPU::add_normals_gpu(const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>&);
template void FrameGPU::add_normals_gpu(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>&);
template void FrameGPU::add_normals_gpu(const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>&);
template void FrameGPU::add_normals_gpu(const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>&);

template void FrameGPU::add_covs(const std::vector<Eigen::Matrix<float, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 3>>>&);
template void FrameGPU::add_covs(const std::vector<Eigen::Matrix<float, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 4>>>&);
template void FrameGPU::add_covs(const std::vector<Eigen::Matrix<double, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 3>>>&);
template void FrameGPU::add_covs(const std::vector<Eigen::Matrix<double, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>>&);
template void FrameGPU::add_covs_gpu(const std::vector<Eigen::Matrix<float, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 3>>>&);
template void FrameGPU::add_covs_gpu(const std::vector<Eigen::Matrix<float, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 4>>>&);
template void FrameGPU::add_covs_gpu(const std::vector<Eigen::Matrix<double, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 3>>>&);
template void FrameGPU::add_covs_gpu(const std::vector<Eigen::Matrix<double, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>>&);

// copy data from GPU to CPU
std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> FrameGPU::get_points_gpu() const {
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points_h;
  if (!points_gpu) {
    return points_h;
  }

  points_h.resize(num_points);
  cudaMemcpy(points_h.data(), points_gpu, sizeof(Eigen::Vector3f) * num_points, cudaMemcpyDeviceToHost);
  return points_h;
}

std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> FrameGPU::get_covs_gpu() const {
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> covs_h;
  if (!covs_gpu) {
    return covs_h;
  }

  covs_h.resize(num_points);
  cudaMemcpy(covs_h.data(), covs_gpu, sizeof(Eigen::Matrix3f) * num_points, cudaMemcpyDeviceToHost);
  return covs_h;
}

}  // namespace gtsam_ext