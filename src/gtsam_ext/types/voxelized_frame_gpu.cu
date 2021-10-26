#include <gtsam_ext/types/voxelized_frame_gpu.hpp>

#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <gtsam_ext/types/cpu_gpu_copy.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_gpu.hpp>

namespace gtsam_ext {

template <typename T, int D>
VoxelizedFrameGPU::VoxelizedFrameGPU(double voxel_resolution, const Eigen::Matrix<T, D, 1>* points, const Eigen::Matrix<T, D, D>* covs, int num_points, bool allocate_cpu)
: times_gpu_storage(new FloatsGPU()),
  points_gpu_storage(new PointsGPU()),
  normals_gpu_storage(new PointsGPU()),
  covs_gpu_storage(new MatricesGPU()) {
  //
  if (allocate_cpu) {
    add_points(points, num_points);
    add_covs(covs, num_points);
  } else {
    add_points_gpu(points, num_points);
    add_covs_gpu(covs, num_points);
  }

  create_voxelmap(voxel_resolution);
}

template <typename T, int D>
VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& points,
  const std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>>& covs,
  bool allocate_cpu)
: VoxelizedFrameGPU(voxel_resolution, points.data(), covs.data(), points.size(), allocate_cpu) {}

VoxelizedFrameGPU::VoxelizedFrameGPU(double voxel_resolution, const Frame& frame, bool allocate_cpu)
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
    copy_to_cpu(times_storage, &times, frame.times, frame.times_gpu, num_points, 0.0);
    copy_to_cpu(points_storage, &points, frame.points, frame.points_gpu, num_points, Eigen::Vector4d(0.0, 0.0, 0.0, 1.0));
    copy_to_cpu(normals_storage, &normals, frame.normals, frame.normals_gpu, num_points, Eigen::Vector4d::Zero().eval());
    copy_to_cpu(covs_storage, &covs, frame.covs, frame.covs_gpu, num_points, Eigen::Matrix4d::Zero().eval());
  }

  create_voxelmap(voxel_resolution);
}

VoxelizedFrameGPU::VoxelizedFrameGPU(double voxel_resolution, const PointsGPU& points, const MatricesGPU& covs, bool allocate_cpu)
: times_gpu_storage(new FloatsGPU()),
  points_gpu_storage(new PointsGPU()),
  normals_gpu_storage(new PointsGPU()),
  covs_gpu_storage(new MatricesGPU()) {
  //
  this->num_points = num_points;
  points_gpu_storage->resize(num_points);
  points_gpu = thrust::raw_pointer_cast(points_gpu_storage->data());
  cudaMemcpy(points_gpu, thrust::raw_pointer_cast(points.data()), sizeof(Eigen::Vector3f) * num_points, cudaMemcpyDeviceToDevice);

  covs_gpu_storage->resize(num_points);
  covs_gpu = thrust::raw_pointer_cast(covs_gpu_storage->data());
  cudaMemcpy(covs_gpu, thrust::raw_pointer_cast(covs.data()), sizeof(Eigen::Matrix3f) * num_points, cudaMemcpyDeviceToDevice);

  if (allocate_cpu) {
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points_h(num_points);
    std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> covs_h(num_points);
    cudaMemcpy(points_h.data(), points_gpu, sizeof(Eigen::Vector3f) * num_points, cudaMemcpyDeviceToHost);
    cudaMemcpy(covs_h.data(), covs_gpu, sizeof(Eigen::Matrix3f) * num_points, cudaMemcpyDeviceToHost);

    points_storage.resize(num_points, Eigen::Vector4d(0.0, 0.0, 0.0, 1.0));
    covs_storage.resize(num_points, Eigen::Matrix4d::Zero());
    this->points = points_storage.data();
    this->covs = covs_storage.data();

    for (int i = 0; i < num_points; i++) {
      this->points[i].head<3>() = points_h[i].cast<double>();
      this->covs[i].block<3, 3>(0, 0) = covs_h[i].cast<double>();
    }
  }

  create_voxelmap(voxel_resolution);
}

VoxelizedFrameGPU::VoxelizedFrameGPU()
: times_gpu_storage(new FloatsGPU()),
  points_gpu_storage(new PointsGPU()),
  normals_gpu_storage(new PointsGPU()),
  covs_gpu_storage(new MatricesGPU()) {}

VoxelizedFrameGPU::~VoxelizedFrameGPU() {}

void VoxelizedFrameGPU::create_voxelmap(double voxel_resolution) {
  if (points && covs) {
    voxels_storage.reset(new GaussianVoxelMapCPU(voxel_resolution));
    voxels_storage->create_voxelmap(*this);
    voxels = voxels_storage;
  }

  voxels_gpu_storage.reset(new GaussianVoxelMapGPU(voxel_resolution));
  voxels_gpu_storage->create_voxelmap(*this);
  voxels_gpu = voxels_gpu_storage;
}

template <typename T>
void VoxelizedFrameGPU::add_times(const T* times, int num_points) {
  add_times_gpu(times, num_points);

  times_storage.resize(num_points);
  thrust::copy(times, times + num_points, times_storage.begin());
  this->times = times_storage.data();
}

template <typename T>
void VoxelizedFrameGPU::add_times_gpu(const T* times, int num_points) {
  assert(num_points == size());
  thrust::host_vector<float> times_h(num_points);
  std::copy(times, times + num_points, times_h.begin());

  times_gpu_storage->resize(num_points);
  cudaMemcpy(thrust::raw_pointer_cast(times_gpu_storage->data()), times_h.data(), sizeof(float) * num_points, cudaMemcpyHostToDevice);
  this->times_gpu = thrust::raw_pointer_cast(times_gpu_storage->data());
}

template <typename T>
void VoxelizedFrameGPU::add_times(const std::vector<T>& times) {
  add_times(times.data(), times.size());
}

template <typename T>
void VoxelizedFrameGPU::add_times_gpu(const std::vector<T>& times) {
  add_times_gpu(times.data(), times.size());
}

template <typename T, int D>
void VoxelizedFrameGPU::add_points(const Eigen::Matrix<T, D, 1>* points, int num_points) {
  add_points_gpu(points, num_points);

  points_storage.resize(num_points, Eigen::Vector4d::UnitW());
  this->points = points_storage.data();
  for (int i = 0; i < num_points; i++) {
    points_storage[i].template head<D>() = points[i].template cast<double>();
  }
}

template <typename T, int D>
void VoxelizedFrameGPU::add_points_gpu(const Eigen::Matrix<T, D, 1>* points, int num_points) {
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
void VoxelizedFrameGPU::add_points(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& points) {
  add_points(points.data(), points.size());
}

template <typename T, int D>
void VoxelizedFrameGPU::add_points_gpu(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& points) {
  add_points_gpu(points.data(), points.size());
}

template <typename T, int D>
void VoxelizedFrameGPU::add_normals(const Eigen::Matrix<T, D, 1>* normals, int num_points) {
  add_normals_gpu(normals, num_points);

  normals_storage.resize(num_points, Eigen::Vector4d(0, 0, 0, 0));
  for (int i = 0; i < num_points; i++) {
    normals_storage[i].template head<D>() = normals[i].template cast<double>();
  }
  this->normals = normals_storage.data();
}

template <typename T, int D>
void VoxelizedFrameGPU::add_normals_gpu(const Eigen::Matrix<T, D, 1>* normals, int num_points) {
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
void VoxelizedFrameGPU::add_normals(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& normals) {
  add_normals(normals.data(), normals.size());
}

template <typename T, int D>
void VoxelizedFrameGPU::add_normals_gpu(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& normals) {
  add_normals_gpu(normals.data(), normals.size());
}

template <typename T, int D>
void VoxelizedFrameGPU::add_covs(const Eigen::Matrix<T, D, D>* covs, int num_points) {
  add_covs_gpu(covs, num_points);

  covs_storage.resize(num_points, Eigen::Matrix4d::Zero());
  for (int i = 0; i < num_points; i++) {
    covs_storage[i].template block<D, D>(0, 0) = covs[i].template cast<double>();
  }
  this->covs = covs_storage.data();
}

template <typename T, int D>
void VoxelizedFrameGPU::add_covs_gpu(const Eigen::Matrix<T, D, D>* covs, int num_points) {
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
void VoxelizedFrameGPU::add_covs(const std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>>& covs) {
  add_covs(covs.data(), covs.size());
}

template <typename T, int D>
void VoxelizedFrameGPU::add_covs_gpu(const std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>>& covs) {
  add_covs_gpu(covs.data(), covs.size());
}

std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> VoxelizedFrameGPU::get_points_gpu() const {
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> buffer(points_gpu_storage->size());
  cudaMemcpy(buffer.data(), thrust::raw_pointer_cast(points_gpu_storage->data()), sizeof(Eigen::Vector3f) * points_gpu_storage->size(), cudaMemcpyDeviceToHost);
  return buffer;
}

std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> VoxelizedFrameGPU::get_covs_gpu() const {
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> buffer(covs_gpu_storage->size());
  cudaMemcpy(buffer.data(), thrust::raw_pointer_cast(covs_gpu_storage->data()), sizeof(Eigen::Matrix3f) * covs_gpu_storage->size(), cudaMemcpyDeviceToHost);
  return buffer;
}

std::vector<int> VoxelizedFrameGPU::get_voxel_num_points() const {
  const auto& num_points_storage = *(voxels_gpu->num_points);
  std::vector<int> buffer(num_points_storage.size());
  cudaMemcpy(buffer.data(), thrust::raw_pointer_cast(num_points_storage.data()), sizeof(int) * num_points_storage.size(), cudaMemcpyDeviceToHost);
  return buffer;
}

std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> VoxelizedFrameGPU::get_voxel_means_gpu() const {
  const auto& means_storage = *(voxels_gpu->voxel_means);
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> buffer(means_storage.size());
  cudaMemcpy(buffer.data(), thrust::raw_pointer_cast(means_storage.data()), sizeof(Eigen::Vector3f) * means_storage.size(), cudaMemcpyDeviceToHost);
  return buffer;
}

std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> VoxelizedFrameGPU::get_voxel_covs_gpu() const {
  const auto& covs_storage = *(voxels_gpu->voxel_covs);
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> buffer(covs_storage.size());
  cudaMemcpy(buffer.data(), thrust::raw_pointer_cast(covs_storage.data()), sizeof(Eigen::Matrix3f) * covs_storage.size(), cudaMemcpyDeviceToHost);
  return buffer;
}

template VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const Eigen::Matrix<float, 3, 1>* points,
  const Eigen::Matrix<float, 3, 3>* covs,
  int num_points,
  bool allocate_cpu);
template VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const Eigen::Matrix<float, 4, 1>* points,
  const Eigen::Matrix<float, 4, 4>* covs,
  int num_points,
  bool allocate_cpu);
template VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const Eigen::Matrix<double, 3, 1>* points,
  const Eigen::Matrix<double, 3, 3>* covs,
  int num_points,
  bool allocate_cpu);
template VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const Eigen::Matrix<double, 4, 1>* points,
  const Eigen::Matrix<double, 4, 4>* covs,
  int num_points,
  bool allocate_cpu);

template VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>&,
  const std::vector<Eigen::Matrix<float, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 3>>>&,
  bool);
template VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>&,
  const std::vector<Eigen::Matrix<float, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 4>>>&,
  bool);
template VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>&,
  const std::vector<Eigen::Matrix<double, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 3>>>&,
  bool);
template VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>&,
  const std::vector<Eigen::Matrix<double, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>>&,
  bool);

template void VoxelizedFrameGPU::add_times(const std::vector<float>&);
template void VoxelizedFrameGPU::add_times(const std::vector<double>&);
template void VoxelizedFrameGPU::add_times_gpu(const std::vector<float>&);
template void VoxelizedFrameGPU::add_times_gpu(const std::vector<double>&);

template void VoxelizedFrameGPU::add_points(const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>&);
template void VoxelizedFrameGPU::add_points(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>&);
template void VoxelizedFrameGPU::add_points(const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>&);
template void VoxelizedFrameGPU::add_points(const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>&);
template void VoxelizedFrameGPU::add_points_gpu(const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>&);
template void VoxelizedFrameGPU::add_points_gpu(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>&);
template void VoxelizedFrameGPU::add_points_gpu(const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>&);
template void VoxelizedFrameGPU::add_points_gpu(const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>&);

template void VoxelizedFrameGPU::add_normals(const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>&);
template void VoxelizedFrameGPU::add_normals(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>&);
template void VoxelizedFrameGPU::add_normals(const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>&);
template void VoxelizedFrameGPU::add_normals(const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>&);
template void VoxelizedFrameGPU::add_normals_gpu(const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>&);
template void VoxelizedFrameGPU::add_normals_gpu(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>&);
template void VoxelizedFrameGPU::add_normals_gpu(const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>&);
template void VoxelizedFrameGPU::add_normals_gpu(const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>&);

template void VoxelizedFrameGPU::add_covs(const std::vector<Eigen::Matrix<float, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 3>>>&);
template void VoxelizedFrameGPU::add_covs(const std::vector<Eigen::Matrix<float, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 4>>>&);
template void VoxelizedFrameGPU::add_covs(const std::vector<Eigen::Matrix<double, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 3>>>&);
template void VoxelizedFrameGPU::add_covs(const std::vector<Eigen::Matrix<double, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>>&);
template void VoxelizedFrameGPU::add_covs_gpu(const std::vector<Eigen::Matrix<float, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 3>>>&);
template void VoxelizedFrameGPU::add_covs_gpu(const std::vector<Eigen::Matrix<float, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 4>>>&);
template void VoxelizedFrameGPU::add_covs_gpu(const std::vector<Eigen::Matrix<double, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 3>>>&);
template void VoxelizedFrameGPU::add_covs_gpu(const std::vector<Eigen::Matrix<double, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>>&);

/**************** merge_voxelized_frames_gpu **********************/

namespace {

struct transform_means_kernel {
  transform_means_kernel(const thrust::device_ptr<const Eigen::Isometry3f>& transform_ptr) : transform_ptr(transform_ptr) {}

  __device__ Eigen::Vector3f operator()(const Eigen::Vector3f& x) const {
    const Eigen::Isometry3f& transform = *thrust::raw_pointer_cast(transform_ptr);
    return transform * x;
  }

  const thrust::device_ptr<const Eigen::Isometry3f> transform_ptr;
};

struct transform_covs_kernel {
  transform_covs_kernel(const thrust::device_ptr<const Eigen::Isometry3f>& transform_ptr) : transform_ptr(transform_ptr) {}

  __device__ Eigen::Matrix3f operator()(const Eigen::Matrix3f& cov) const {
    const Eigen::Isometry3f& transform = *thrust::raw_pointer_cast(transform_ptr);
    return transform.linear() * cov * transform.linear().transpose();
  }

  const thrust::device_ptr<const Eigen::Isometry3f> transform_ptr;
};
}  // namespace

VoxelizedFrame::Ptr merge_voxelized_frames_gpu(
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& poses,
  const std::vector<Frame::ConstPtr>& frames,
  double downsample_resolution,
  double voxel_resolution,
  bool allocate_cpu) {
  //
  int num_all_points = 0;
  thrust::host_vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>> h_poses(poses.size());
  for (int i = 0; i < poses.size(); i++) {
    h_poses[i] = poses[i].cast<float>();
    num_all_points += frames[i]->size();
  }

  thrust::device_vector<Eigen::Isometry3f> d_poses(h_poses);
  cudaMemcpy(thrust::raw_pointer_cast(d_poses.data()), thrust::raw_pointer_cast(h_poses.data()), sizeof(Eigen::Isometry3f) * h_poses.size(), cudaMemcpyHostToDevice);

  thrust::device_vector<Eigen::Vector3f> all_points(num_all_points);
  thrust::device_vector<Eigen::Matrix3f> all_covs(num_all_points);

  size_t begin = 0;
  for (int i = 0; i < frames.size(); i++) {
    const auto& frame = frames[i];
    const thrust::device_ptr<const Eigen::Isometry3f> transform_ptr(d_poses.data() + i);

    const thrust::device_ptr<const Eigen::Vector3f> points_ptr(frame->points_gpu);
    const thrust::device_ptr<const Eigen::Matrix3f> covs_ptr(frame->covs_gpu);

    thrust::transform(points_ptr, points_ptr + frame->size(), all_points.begin() + begin, transform_means_kernel(transform_ptr));
    thrust::transform(covs_ptr, covs_ptr + frame->size(), all_covs.begin() + begin, transform_covs_kernel(transform_ptr));
    begin += frame->size();
  }

  Frame all_frames;
  all_frames.num_points = num_all_points;
  all_frames.points_gpu = thrust::raw_pointer_cast(all_points.data());
  all_frames.covs_gpu = thrust::raw_pointer_cast(all_covs.data());

  GaussianVoxelMapGPU downsampling(downsample_resolution, num_all_points / 10);
  downsampling.create_voxelmap(all_frames);

  return VoxelizedFrame::Ptr(new VoxelizedFrameGPU(voxel_resolution, *downsampling.voxel_means, *downsampling.voxel_covs, allocate_cpu));
}

}  // namespace gtsam_ext