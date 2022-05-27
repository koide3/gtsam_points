// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_ext/types/voxelized_frame_gpu.hpp>

#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/async/reduce.h>
#include <thrust/async/transform.h>

#include <gtsam_ext/types/cpu_gpu_copy.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_gpu.hpp>
#include <gtsam_ext/cuda/kernels/vector3_hash.cuh>

namespace gtsam_ext {

// constructor with points & covs
template <typename T, int D>
VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const Eigen::Matrix<T, D, 1>* points,
  const Eigen::Matrix<T, D, D>* covs,
  int num_points,
  bool allocate_cpu)
: times_gpu_storage(new FloatsGPU()),
  points_gpu_storage(new PointsGPU()),
  normals_gpu_storage(new PointsGPU()),
  covs_gpu_storage(new MatricesGPU()),
  intensities_gpu_storage(new FloatsGPU()) {
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

// deep copy constructor
VoxelizedFrameGPU::VoxelizedFrameGPU(double voxel_resolution, const Frame& frame, bool allocate_cpu)
: times_gpu_storage(new FloatsGPU()),
  points_gpu_storage(new PointsGPU()),
  normals_gpu_storage(new PointsGPU()),
  covs_gpu_storage(new MatricesGPU()),
  intensities_gpu_storage(new FloatsGPU()) {
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

    if (frame.intensities) {
      add_intensities(frame.intensities, frame.size());
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

    if (frame.intensities) {
      add_intensities_gpu(frame.intensities, frame.size());
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

  create_voxelmap(voxel_resolution);
}

VoxelizedFrameGPU::VoxelizedFrameGPU(double voxel_resolution, const PointsGPU& points, const MatricesGPU& covs, bool allocate_cpu)
: times_gpu_storage(new FloatsGPU()),
  points_gpu_storage(new PointsGPU()),
  normals_gpu_storage(new PointsGPU()),
  covs_gpu_storage(new MatricesGPU()),
  intensities_gpu_storage(new FloatsGPU()) {
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
  covs_gpu_storage(new MatricesGPU()),
  intensities_gpu_storage(new FloatsGPU()) {}

VoxelizedFrameGPU::~VoxelizedFrameGPU() {}

void VoxelizedFrameGPU::create_voxelmap(double voxel_resolution) {
  if (points && covs) {
    voxels_storage.reset(new GaussianVoxelMapCPU(voxel_resolution));
    voxels_storage->insert(*this);
    voxels = voxels_storage;
  }

  voxels_gpu_storage.reset(new GaussianVoxelMapGPU(voxel_resolution));
  voxels_gpu_storage->insert(*this);
  voxels_gpu = voxels_gpu_storage;
}

// add_times
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

template void VoxelizedFrameGPU::add_times(const float* times, int num_points);
template void VoxelizedFrameGPU::add_times(const double* times, int num_points);

// add_points
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

template void VoxelizedFrameGPU::add_points(const Eigen::Matrix<float, 3, 1>* points, int num_points);
template void VoxelizedFrameGPU::add_points(const Eigen::Matrix<float, 4, 1>* points, int num_points);
template void VoxelizedFrameGPU::add_points(const Eigen::Matrix<double, 3, 1>* points, int num_points);
template void VoxelizedFrameGPU::add_points(const Eigen::Matrix<double, 4, 1>* points, int num_points);

// add_normals
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
  assert(num_points == size());

  thrust::host_vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> normals_h(num_points);
  for (int i = 0; i < num_points; i++) {
    normals_h[i] = normals[i].template head<3>().template cast<float>();
  }

  normals_gpu_storage->resize(num_points);
  cudaMemcpy(thrust::raw_pointer_cast(normals_gpu_storage->data()), normals_h.data(), sizeof(Eigen::Vector3f) * num_points, cudaMemcpyHostToDevice);
  this->normals_gpu = thrust::raw_pointer_cast(normals_gpu_storage->data());
}

template void VoxelizedFrameGPU::add_normals(const Eigen::Matrix<float, 3, 1>* normals, int num_points);
template void VoxelizedFrameGPU::add_normals(const Eigen::Matrix<float, 4, 1>* normals, int num_points);
template void VoxelizedFrameGPU::add_normals(const Eigen::Matrix<double, 3, 1>* normals, int num_points);
template void VoxelizedFrameGPU::add_normals(const Eigen::Matrix<double, 4, 1>* normals, int num_points);

// add_covs
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
  assert(num_points == size());
  thrust::host_vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> covs_h(num_points);
  for (int i = 0; i < num_points; i++) {
    covs_h[i] = covs[i].template block<3, 3>(0, 0).template cast<float>();
  }

  covs_gpu_storage->resize(num_points);
  cudaMemcpy(thrust::raw_pointer_cast(covs_gpu_storage->data()), covs_h.data(), sizeof(Eigen::Matrix3f) * num_points, cudaMemcpyHostToDevice);
  this->covs_gpu = thrust::raw_pointer_cast(covs_gpu_storage->data());
}

template void VoxelizedFrameGPU::add_covs(const Eigen::Matrix<float, 3, 3>* covs, int num_points);
template void VoxelizedFrameGPU::add_covs(const Eigen::Matrix<float, 4, 4>* covs, int num_points);
template void VoxelizedFrameGPU::add_covs(const Eigen::Matrix<double, 3, 3>* covs, int num_points);
template void VoxelizedFrameGPU::add_covs(const Eigen::Matrix<double, 4, 4>* covs, int num_points);

// add_intensities
template <typename T>
void VoxelizedFrameGPU::add_intensities(const T* intensities, int num_points) {
  add_intensities_gpu(intensities, num_points);

  intensities_storage.resize(num_points);
  thrust::copy(intensities, intensities + num_points, intensities_storage.begin());
  this->intensities = intensities_storage.data();
}
template <typename T>
void VoxelizedFrameGPU::add_intensities_gpu(const T* intensities, int num_points) {
  assert(num_points == size());
  thrust::host_vector<float> intensities_h(num_points);
  std::copy(intensities, intensities + num_points, intensities_h.begin());

  intensities_gpu_storage->resize(num_points);
  cudaMemcpy(thrust::raw_pointer_cast(intensities_gpu_storage->data()), intensities_h.data(), sizeof(float) * num_points, cudaMemcpyHostToDevice);
  this->intensities_gpu = thrust::raw_pointer_cast(intensities_gpu_storage->data());
}

template void VoxelizedFrameGPU::add_intensities(const float* intensities, int num_points);
template void VoxelizedFrameGPU::add_intensities(const double* intensities, int num_points);

// GPU-to-CPU copy
std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> VoxelizedFrameGPU::get_points_gpu() const {
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> buffer(points_gpu_storage->size());
  cudaMemcpy(
    buffer.data(),
    thrust::raw_pointer_cast(points_gpu_storage->data()),
    sizeof(Eigen::Vector3f) * points_gpu_storage->size(),
    cudaMemcpyDeviceToHost);
  return buffer;
}

std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> VoxelizedFrameGPU::get_covs_gpu() const {
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> buffer(covs_gpu_storage->size());
  cudaMemcpy(
    buffer.data(),
    thrust::raw_pointer_cast(covs_gpu_storage->data()),
    sizeof(Eigen::Matrix3f) * covs_gpu_storage->size(),
    cudaMemcpyDeviceToHost);
  return buffer;
}

std::vector<std::pair<Eigen::Vector3i, int>> VoxelizedFrameGPU::get_voxel_buckets_gpu() const {
  const auto& buckets_storage = *(voxels_gpu->buckets);
  std::vector<std::pair<Eigen::Vector3i, int>> buffer(buckets_storage.size());
  cudaMemcpy(
    buffer.data(),
    thrust::raw_pointer_cast(buckets_storage.data()),
    sizeof(thrust::pair<Eigen::Vector3i, int>) * buckets_storage.size(),
    cudaMemcpyDeviceToHost);
  return buffer;
}

std::vector<int> VoxelizedFrameGPU::get_voxel_num_points_gpu() const {
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
  cudaMemcpy(
    thrust::raw_pointer_cast(d_poses.data()),
    thrust::raw_pointer_cast(h_poses.data()),
    sizeof(Eigen::Isometry3f) * h_poses.size(),
    cudaMemcpyHostToDevice);

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
  downsampling.insert(all_frames);

  return VoxelizedFrame::Ptr(new VoxelizedFrameGPU(voxel_resolution, *downsampling.voxel_means, *downsampling.voxel_covs, allocate_cpu));
}

namespace {
// point coord -> voxel coord conversion
struct overlap_count_kernel {
public:
  overlap_count_kernel(const GaussianVoxelMapGPU& voxelmap, const thrust::device_ptr<const Eigen::Isometry3f>& delta_ptr)
  : voxelmap_info_ptr(voxelmap.voxelmap_info_ptr->data()),
    buckets_ptr(voxelmap.buckets->data()),
    delta_ptr(delta_ptr) {}

  __host__ __device__ bool operator()(const Eigen::Vector3f& x) const {
    const auto& info = *thrust::raw_pointer_cast(voxelmap_info_ptr);
    const auto& trans = *thrust::raw_pointer_cast(delta_ptr);

    Eigen::Vector3f x_ = trans.linear() * x + trans.translation();
    int voxel_index = lookup_voxel(info.max_bucket_scan_count, info.num_buckets, buckets_ptr, info.voxel_resolution, x_);

    return voxel_index >= 0;
  }

  thrust::device_ptr<const VoxelMapInfo> voxelmap_info_ptr;
  thrust::device_ptr<const thrust::pair<Eigen::Vector3i, int>> buckets_ptr;

  thrust::device_ptr<const Eigen::Isometry3f> delta_ptr;
};

struct bool_or_kernel {
  __host__ __device__ bool operator()(const thrust::tuple<bool, bool>& b) const { return thrust::get<0>(b) || thrust::get<1>(b); }
};

template <typename T_DST>
struct cast_kernel {
  template <typename T_SRC>
  __host__ __device__ T_DST operator()(const T_SRC& value) const {
    return static_cast<T_DST>(value);
  }
};
}  // namespace

double overlap_gpu(const GaussianVoxelMap::ConstPtr& target_, const Frame::ConstPtr& source, const Eigen::Isometry3f* delta_gpu) {
  if (!source->points_gpu) {
    std::cerr << "error: GPU source points have not been allocated!!" << std::endl;
    abort();
  }

  auto target = std::dynamic_pointer_cast<const GaussianVoxelMapGPU>(target_);
  if (!target) {
    std::cerr << "error: Failed to cast target voxelmap to GaussianVoxelMapGPU!!" << std::endl;
    abort();
  }

  thrust::device_vector<bool> overlap(source->size());
  auto trans_result = thrust::async::transform(
    thrust::device_ptr<Eigen::Vector3f>(source->points_gpu),
    thrust::device_ptr<Eigen::Vector3f>(source->points_gpu) + source->size(),
    overlap.begin(),
    overlap_count_kernel(*target, thrust::device_ptr<const Eigen::Isometry3f>(delta_gpu)));
  auto reduce_result = thrust::async::reduce(
    thrust::cuda::par.after(trans_result),
    thrust::make_transform_iterator(overlap.begin(), cast_kernel<int>()),
    thrust::make_transform_iterator(overlap.end(), cast_kernel<int>()),
    0);

  int num_inliers = reduce_result.get();

  return static_cast<double>(num_inliers) / source->size();
}

double overlap_gpu(const VoxelizedFrame::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3f* delta_gpu) {
  return overlap_gpu(target->voxels_gpu, source, delta_gpu);
}

double overlap_gpu(const GaussianVoxelMap::ConstPtr& target_, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta) {
  if (!source->points_gpu) {
    std::cerr << "error: GPU source points have not been allocated!!" << std::endl;
    abort();
  }

  auto target = std::dynamic_pointer_cast<const GaussianVoxelMapGPU>(target_);
  if (!target) {
    std::cerr << "error: Failed to cast target voxelmap to GaussianVoxelMapGPU!!" << std::endl;
    abort();
  }

  thrust::host_vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>> deltas(1);
  deltas[0] = delta.cast<float>();
  thrust::device_vector<Eigen::Isometry3f> delta_ptr = deltas;

  return overlap_gpu(target, source, thrust::raw_pointer_cast(delta_ptr.data()));
}

double overlap_gpu(const VoxelizedFrame::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta) {
  return overlap_gpu(target->voxels_gpu, source, delta);
}

double overlap_gpu(
  const std::vector<GaussianVoxelMap::ConstPtr>& targets_,
  const Frame::ConstPtr& source,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas_) {
  if (!source->points_gpu) {
    std::cerr << "error: GPU source points have not been allocated!!" << std::endl;
    abort();
  }

  std::vector<GaussianVoxelMapGPU::ConstPtr> targets(targets_.size());
  for (int i = 0; i < targets_.size(); i++) {
    targets[i] = std::dynamic_pointer_cast<const GaussianVoxelMapGPU>(targets_[i]);
    if (!targets[i]) {
      std::cerr << "error: Failed to cast target voxelmap to GaussianVoxelMapGPU!!" << std::endl;
    }
  }

  thrust::host_vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>> deltas(deltas_.size());
  thrust::transform(deltas_.begin(), deltas_.end(), deltas.begin(), [](const Eigen::Isometry3d& delta) { return delta.cast<float>(); });
  thrust::device_vector<Eigen::Isometry3f> deltas_ptr = deltas;

  thrust::device_vector<bool> overlap(source->size(), false);
  std::vector<thrust::system::cuda::unique_eager_event> results(targets.size());

  for (int i = 0; i < targets.size(); i++) {
    overlap_count_kernel overlap_kernel(*targets[i], deltas_ptr.data() + i);
    auto first = thrust::make_transform_iterator(thrust::device_ptr<Eigen::Vector3f>(source->points_gpu), overlap_kernel);
    auto last = thrust::make_transform_iterator(thrust::device_ptr<Eigen::Vector3f>(source->points_gpu) + source->size(), overlap_kernel);

    if (i == 0) {
      auto result = thrust::async::transform(
        thrust::make_zip_iterator(thrust::make_tuple(overlap.begin(), first)),
        thrust::make_zip_iterator(thrust::make_tuple(overlap.end(), last)),
        overlap.begin(),
        bool_or_kernel());

      results[i] = std::move(result);
    } else {
      auto result = thrust::async::transform(
        thrust::cuda::par.after(results[i - 1]),
        thrust::make_zip_iterator(thrust::make_tuple(overlap.begin(), first)),
        thrust::make_zip_iterator(thrust::make_tuple(overlap.end(), last)),
        overlap.begin(),
        bool_or_kernel());

      results[i] = std::move(result);
    }
  }

  auto result = thrust::async::reduce(
    thrust::cuda::par.after(results.back()),
    thrust::make_transform_iterator(overlap.begin(), cast_kernel<int>()),
    thrust::make_transform_iterator(overlap.end(), cast_kernel<int>()),
    0);
  int num_inliers = result.get();

  return static_cast<double>(num_inliers) / source->size();
}

double overlap_gpu(
  const std::vector<VoxelizedFrame::ConstPtr>& targets,
  const Frame::ConstPtr& source,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas_) {
  std::vector<GaussianVoxelMap::ConstPtr> target_voxelmaps(targets.size());
  std::transform(targets.begin(), targets.end(), target_voxelmaps.begin(), [](const VoxelizedFrame::ConstPtr& frame) { return frame->voxels_gpu; });
  return overlap_gpu(target_voxelmaps, source, deltas_);
}

}  // namespace gtsam_ext