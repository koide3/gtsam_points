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

namespace {

template <typename T_GPU, typename T_CPU>
T_GPU cast_to_gpu(const T_CPU& v) {
  return v.template block<T_GPU::RowsAtCompileTime, T_GPU::ColsAtCompileTime>(0, 0).template cast<float>();
}

template <>
float cast_to_gpu(const double& v) {
  return static_cast<float>(v);
}

template <typename T_CPU, typename T_GPU>
T_CPU cast_to_cpu(const T_GPU& v) {
  T_CPU v_cpu = T_CPU::Zero();
  if constexpr (T_CPU::ColsAtCompileTime == 1) {
    v_cpu[T_CPU::ColsAtCompileTime - 1] = 1.0;
  }
  v_cpu.template block<T_GPU::RowsAtCompileTime, T_GPU::ColsAtCompileTime>(0, 0) = v.template cast<double>();
  return v_cpu;
}

template <>
double cast_to_cpu(const float& v) {
  return static_cast<double>(v);
}

template <typename T_GPU, typename T_CPU>
void copy_to_gpu(thrust::device_vector<T_GPU>& output_storate, T_GPU** output_gpu, const T_CPU* input_cpu, const T_GPU* input_gpu, int num_points) {
  if (!input_cpu && !input_gpu) {
    return;
  }

  output_storate.resize(num_points);
  *output_gpu = thrust::raw_pointer_cast(output_storate.data());

  if (input_gpu) {
    cudaMemcpy(*output_gpu, input_gpu, sizeof(T_GPU) * num_points, cudaMemcpyDeviceToDevice);
  } else {
    std::vector<T_GPU, Eigen::aligned_allocator<T_GPU>> floats(num_points);
    std::transform(input_cpu, input_cpu + num_points, floats.begin(), [](const T_CPU& v) { return cast_to_gpu<T_GPU, T_CPU>(v); });
    cudaMemcpy(*output_gpu, floats.data(), sizeof(T_GPU) * num_points, cudaMemcpyHostToDevice);
  }
}

template <typename Vector, typename T_CPU, typename T_GPU>
void copy_to_cpu(Vector& output_storage, T_CPU** output_cpu, const T_CPU* input_cpu, const T_GPU* input_gpu, int num_points) {
  if (!input_cpu && !input_gpu) {
    return;
  }

  output_storage.resize(num_points);
  *output_cpu = output_storage.data();

  if (input_cpu) {
    memcpy(*output_cpu, input_cpu, sizeof(T_CPU) * num_points);
  } else {
    std::vector<T_GPU, Eigen::aligned_allocator<T_GPU>> floats(num_points);
    cudaMemcpy(floats.data(), input_gpu, sizeof(T_GPU) * num_points, cudaMemcpyDeviceToHost);
    std::transform(floats.begin(), floats.end(), *output_cpu, [](const T_GPU& v) { return cast_to_cpu<T_CPU, T_GPU>(v); });
  }
}

}  // namespace

FrameGPU::FrameGPU(const Frame& frame, bool allocate_cpu)
: times_gpu_storage(new FloatsGPU()),
  points_gpu_storage(new PointsGPU()),
  normals_gpu_storage(new PointsGPU()),
  covs_gpu_storage(new MatricesGPU()) {
  //
  num_points = frame.size();

  copy_to_gpu(*times_gpu_storage, &times_gpu, frame.times, frame.times_gpu, num_points);
  copy_to_gpu(*points_gpu_storage, &points_gpu, frame.points, frame.points_gpu, num_points);
  copy_to_gpu(*normals_gpu_storage, &normals_gpu, frame.normals, frame.normals_gpu, num_points);
  copy_to_gpu(*covs_gpu_storage, &covs_gpu, frame.covs, frame.covs_gpu, num_points);

  if (allocate_cpu) {
    copy_to_cpu(times_storage, &times, frame.times, frame.times_gpu, num_points);
    copy_to_cpu(points_storage, &points, frame.points, frame.points_gpu, num_points);
    copy_to_cpu(normals_storage, &normals, frame.normals, frame.normals_gpu, num_points);
    copy_to_cpu(covs_storage, &covs, frame.covs, frame.covs_gpu, num_points);
  }
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