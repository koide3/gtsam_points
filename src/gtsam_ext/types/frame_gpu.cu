#include <gtsam_ext/types/frame_gpu.hpp>

#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace gtsam_ext {

template <typename T, int D>
FrameGPU::FrameGPU(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& points, bool allocate_cpu)
: times_gpu_storage(new FloatsGPU()),
  points_gpu_storage(new PointsGPU()),
  normals_gpu_storage(new PointsGPU()),
  covs_gpu_storage(new MatricesGPU()) {
  //
  if (allocate_cpu) {
    points_storage.resize(points.size(), Eigen::Vector4d(0, 0, 0, 1));
    for (int i = 0; i < points.size(); i++) {
      points_storage[i].template head<D>() = points[i].template cast<double>();
    }
    this->points = points_storage.data();
  }

  thrust::host_vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>> points_h(points.size());
  for (int i = 0; i < points.size(); i++) {
    points_h[i] = points[i].template head<3>().template cast<float>();
  }

  points_gpu_storage->resize(points.size());
  cudaMemcpy(thrust::raw_pointer_cast(points_gpu_storage->data()), points_h.data(), sizeof(Eigen::Vector3f) * points.size(), cudaMemcpyHostToDevice);
  this->points_gpu = thrust::raw_pointer_cast(points_gpu_storage->data());
  this->num_points = points.size();
}

FrameGPU::~FrameGPU() {}

template <typename T>
void FrameGPU::add_times(const std::vector<T>& times) {
  assert(times.size() == size());
  times_storage.resize(times.size());
  thrust::transform(times.begin(), times.end(), times_storage.begin(), [](const auto& v) { return static_cast<double>(v); });
  this->times = times_storage.data();

  add_times_gpu(times);
}

template <typename T>
void FrameGPU::add_times_gpu(const std::vector<T>& times) {
  assert(times.size() == size());

  thrust::host_vector<float> times_h(times.size());
  std::transform(times.begin(), times.end(), times_h.begin(), [](const auto& v) { return static_cast<float>(v); });

  times_gpu_storage->resize(times.size());
  cudaMemcpy(thrust::raw_pointer_cast(times_gpu_storage->data()), times_h.data(), sizeof(float) * times.size(), cudaMemcpyHostToDevice);
  this->times_gpu = thrust::raw_pointer_cast(times_gpu_storage->data());
}

template <typename T, int D>
void FrameGPU::add_normals(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& normals) {
  assert(normals.size() == size());
  normals_storage.resize(normals.size(), Eigen::Vector4d(0, 0, 0, 0));
  for (int i = 0; i < normals.size(); i++) {
    normals_storage[i].template head<D>() = normals[i].template cast<double>();
  }
  this->normals = normals_storage.data();

  add_normals_gpu(normals);
}

template <typename T, int D>
void FrameGPU::add_normals_gpu(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& normals) {
  assert(normals.size() == size());

  thrust::host_vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> normals_h(normals.size());
  for (int i = 0; i < normals.size(); i++) {
    normals_h[i] = normals[i].template head<3>().template cast<float>();
  }

  normals_gpu_storage->resize(normals.size());
  cudaMemcpy(thrust::raw_pointer_cast(normals_gpu_storage->data()), normals_h.data(), sizeof(Eigen::Vector3f) * normals.size(), cudaMemcpyHostToDevice);
  this->normals_gpu = thrust::raw_pointer_cast(normals_gpu_storage->data());
}

template <typename T, int D>
void FrameGPU::add_covs(const std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>>& covs) {
  assert(covs.size() == size());
  covs_storage.resize(covs.size(), Eigen::Matrix4d::Zero());
  for (int i = 0; i < covs.size(); i++) {
    covs_storage[i].template block<D, D>(0, 0) = covs[i].template cast<double>();
  }
  this->covs = covs_storage.data();

  add_covs_gpu(covs);
}

template <typename T, int D>
void FrameGPU::add_covs_gpu(const std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>>& covs) {
  assert(covs.size() == size());
  thrust::host_vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> covs_h(covs.size());
  for (int i = 0; i < covs.size(); i++) {
    covs_h[i] = covs[i].template block<3, 3>(0, 0).template cast<float>();
  }

  covs_gpu_storage->resize(covs.size());
  cudaMemcpy(thrust::raw_pointer_cast(covs_gpu_storage->data()), covs_h.data(), sizeof(Eigen::Matrix3f) * covs.size(), cudaMemcpyHostToDevice);
  this->covs_gpu = thrust::raw_pointer_cast(covs_gpu_storage->data());
}

template FrameGPU::FrameGPU(const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>&, bool);
template FrameGPU::FrameGPU(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>&, bool);
template FrameGPU::FrameGPU(const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>&, bool);
template FrameGPU::FrameGPU(const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>&, bool);

template void FrameGPU::add_times(const std::vector<float>&);
template void FrameGPU::add_times(const std::vector<double>&);

template void FrameGPU::add_normals(const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>&);
template void FrameGPU::add_normals(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>&);
template void FrameGPU::add_normals(const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>&);
template void FrameGPU::add_normals(const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>&);

template void FrameGPU::add_covs(const std::vector<Eigen::Matrix<float, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 3>>>&);
template void FrameGPU::add_covs(const std::vector<Eigen::Matrix<float, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 4>>>&);
template void FrameGPU::add_covs(const std::vector<Eigen::Matrix<double, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 3>>>&);
template void FrameGPU::add_covs(const std::vector<Eigen::Matrix<double, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>>&);

}  // namespace gtsam_ext