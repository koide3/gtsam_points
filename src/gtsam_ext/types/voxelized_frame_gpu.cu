// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_ext/types/voxelized_frame_gpu.hpp>

#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/async/reduce.h>
#include <thrust/async/transform.h>

#include <gtsam_ext/types/cpu_gpu_copy.hpp>
#include <gtsam_ext/types/voxelized_frame_cpu.hpp>
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
  int num_points) {
  //
  FrameGPU::add_points(points, num_points);
  FrameGPU::add_covs(covs, num_points);

  create_voxelmap(voxel_resolution);
}

template VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const Eigen::Matrix<float, 3, 1>* points,
  const Eigen::Matrix<float, 3, 3>* covs,
  int num_points);
template VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const Eigen::Matrix<float, 4, 1>* points,
  const Eigen::Matrix<float, 4, 4>* covs,
  int num_points);
template VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const Eigen::Matrix<double, 3, 1>* points,
  const Eigen::Matrix<double, 3, 3>* covs,
  int num_points);
template VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const Eigen::Matrix<double, 4, 1>* points,
  const Eigen::Matrix<double, 4, 4>* covs,
  int num_points);

// deep copy constructor
VoxelizedFrameGPU::VoxelizedFrameGPU(double voxel_resolution, const Frame& frame) : FrameGPU(frame) {
  create_voxelmap(voxel_resolution);
}

VoxelizedFrameGPU::VoxelizedFrameGPU() {}
VoxelizedFrameGPU::~VoxelizedFrameGPU() {}

void VoxelizedFrameGPU::create_voxelmap(double voxel_resolution) {
  if (!has_points() || !has_covs()) {
    std::cerr << "error: frame does not have points or covs!!" << std::endl;
  }

  voxels.reset(new GaussianVoxelMapCPU(voxel_resolution));
  voxels->insert(*this);

  create_voxelmap_gpu(voxel_resolution);
}

void VoxelizedFrameGPU::create_voxelmap_gpu(double voxel_resolution) {
  if (!points_gpu || !covs_gpu) {
    std::cerr << "error: frame does not have points or covs!!" << std::endl;
  }

  voxels_gpu.reset(new GaussianVoxelMapGPU(voxel_resolution));
  voxels_gpu->insert(*this);
}

// GPU-to-CPU copy
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

namespace {

struct transform_means_kernel {
  transform_means_kernel(const thrust::device_ptr<const Eigen::Isometry3f>& transform_ptr) : transform_ptr(transform_ptr) {}

  __device__ Eigen::Vector3f operator()(const Eigen::Vector3f& x) const {
    const Eigen::Isometry3f& transform = *thrust::raw_pointer_cast(transform_ptr);
    return transform.linear() * x + transform.translation();
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

Frame::Ptr merge_frames_gpu(
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& poses,
  const std::vector<Frame::ConstPtr>& frames,
  double downsample_resolution,
  double voxel_resolution,
  CUstream_st* stream) {
  //
  int num_all_points = 0;
  std::vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>> h_poses(poses.size());
  for (int i = 0; i < poses.size(); i++) {
    h_poses[i] = poses[i].cast<float>();
    num_all_points += frames[i]->size();
  }

  Eigen::Isometry3f* d_poses;
  cudaMallocAsync(&d_poses, sizeof(Eigen::Isometry3f) * poses.size(), stream);
  cudaMemcpyAsync(d_poses, h_poses.data(), sizeof(Eigen::Isometry3f) * poses.size(), cudaMemcpyHostToDevice, stream);

  Eigen::Vector3f* all_points;
  Eigen::Matrix3f* all_covs;
  cudaMallocAsync(&all_points, sizeof(Eigen::Vector3f) * num_all_points, stream);
  cudaMallocAsync(&all_covs, sizeof(Eigen::Matrix3f) * num_all_points, stream);

  const thrust::device_ptr<Eigen::Vector3f> all_points_ptr(all_points);
  const thrust::device_ptr<Eigen::Matrix3f> all_covs_ptr(all_covs);

  std::vector<thrust::system::cuda::unique_eager_event> results(frames.size());

  size_t begin = 0;
  for (int i = 0; i < frames.size(); i++) {
    const auto& frame = frames[i];
    const thrust::device_ptr<const Eigen::Isometry3f> transform_ptr(d_poses + i);
    const thrust::device_ptr<const Eigen::Vector3f> points_ptr(frame->points_gpu);
    const thrust::device_ptr<const Eigen::Matrix3f> covs_ptr(frame->covs_gpu);

    results[i] = thrust::async::transform(
      thrust::cuda::par.on(stream),
      points_ptr,
      points_ptr + frame->size(),
      all_points_ptr + begin,
      transform_means_kernel(transform_ptr));
    thrust::transform(thrust::cuda::par.on(stream), covs_ptr, covs_ptr + frame->size(), all_covs_ptr + begin, transform_covs_kernel(transform_ptr));
    begin += frame->size();
  }

  cudaStreamSynchronize(stream);
  for (auto& result : results) {
    result.wait();
  }

  Frame all_frames;
  all_frames.num_points = num_all_points;
  all_frames.points_gpu = all_points;
  all_frames.covs_gpu = all_covs;

  GaussianVoxelMapGPU downsampling(downsample_resolution, num_all_points / 10);
  downsampling.insert(all_frames);

  const int num_voxels = downsampling.voxel_means->size();
  const Eigen::Vector3f* voxel_means = thrust::raw_pointer_cast(downsampling.voxel_means->data());
  const Eigen::Matrix3f* voxel_covs = thrust::raw_pointer_cast(downsampling.voxel_covs->data());

  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> means(num_voxels);
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> covs(num_voxels);

  cudaMemcpyAsync(means.data(), voxel_means, sizeof(Eigen::Vector3f) * num_voxels, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(covs.data(), voxel_covs, sizeof(Eigen::Matrix3f) * num_voxels, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cudaFreeHost(d_poses);
  cudaFreeHost(all_points);
  cudaFreeHost(all_covs);

  return std::make_shared<VoxelizedFrameGPU>(voxel_resolution, means, covs);
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

double
overlap_gpu(const GaussianVoxelMap::ConstPtr& target_, const Frame::ConstPtr& source, const Eigen::Isometry3f* delta_gpu, CUstream_st* stream) {
  if (!source->points_gpu) {
    std::cerr << "error: GPU source points have not been allocated!!" << std::endl;
    abort();
  }

  auto target = std::dynamic_pointer_cast<const GaussianVoxelMapGPU>(target_);
  if (!target) {
    std::cerr << "error: Failed to cast target voxelmap to GaussianVoxelMapGPU!!" << std::endl;
    abort();
  }

  bool* overlap;
  cudaMallocAsync(&overlap, sizeof(bool) * source->size(), stream);
  thrust::device_ptr<bool> overlap_ptr(overlap);

  auto trans_result = thrust::async::transform(
    thrust::cuda::par.on(stream),
    thrust::device_ptr<Eigen::Vector3f>(source->points_gpu),
    thrust::device_ptr<Eigen::Vector3f>(source->points_gpu) + source->size(),
    overlap_ptr,
    overlap_count_kernel(*target, thrust::device_ptr<const Eigen::Isometry3f>(delta_gpu)));
  auto reduce_result = thrust::async::reduce(
    thrust::cuda::par.after(trans_result),
    thrust::make_transform_iterator(overlap_ptr, cast_kernel<int>()),
    thrust::make_transform_iterator(overlap_ptr, cast_kernel<int>()),
    0);

  int num_inliers = reduce_result.get();

  cudaFreeAsync(&overlap, stream);

  return static_cast<double>(num_inliers) / source->size();
}

double overlap_gpu(const Frame::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3f* delta_gpu, CUstream_st* stream) {
  return overlap_gpu(target->voxels_gpu, source, delta_gpu, stream);
}

double overlap_gpu(const GaussianVoxelMap::ConstPtr& target_, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta, CUstream_st* stream) {
  if (!source->points_gpu) {
    std::cerr << "error: GPU source points have not been allocated!!" << std::endl;
    abort();
  }

  auto target = std::dynamic_pointer_cast<const GaussianVoxelMapGPU>(target_);
  if (!target) {
    std::cerr << "error: Failed to cast target voxelmap to GaussianVoxelMapGPU!!" << std::endl;
    abort();
  }

  Eigen::Isometry3f h_delta = delta.cast<float>();
  Eigen::Isometry3f* d_delta;
  cudaMallocAsync(&d_delta, sizeof(Eigen::Isometry3f), stream);
  cudaMemcpyAsync(d_delta, h_delta.data(), sizeof(Eigen::Isometry3f), cudaMemcpyHostToDevice, stream);

  double overlap = overlap_gpu(target, source, d_delta, stream);

  cudaFreeAsync(d_delta, stream);

  return overlap;
}

double overlap_gpu(const Frame::ConstPtr& target, const Frame::ConstPtr& source, const Eigen::Isometry3d& delta, CUstream_st* stream) {
  return overlap_gpu(target->voxels_gpu, source, delta, stream);
}

double overlap_gpu(
  const std::vector<GaussianVoxelMap::ConstPtr>& targets_,
  const Frame::ConstPtr& source,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas_,
  CUstream_st* stream) {
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

  std::vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>> deltas(deltas_.size());
  std::transform(deltas_.begin(), deltas_.end(), deltas.begin(), [](const Eigen::Isometry3d& delta) { return delta.cast<float>(); });

  Eigen::Isometry3f* d_deltas;
  cudaMallocAsync(&d_deltas, sizeof(Eigen::Isometry3f) * deltas.size(), stream);
  cudaMemcpyAsync(d_deltas, deltas.data(), sizeof(Eigen::Isometry3f) * deltas.size(), cudaMemcpyHostToDevice, stream);
  thrust::device_ptr<Eigen::Isometry3f> deltas_ptr(d_deltas);

  bool* d_overlap;
  cudaMallocAsync(&d_overlap, sizeof(bool) * source->size(), stream);
  cudaMemsetAsync(d_overlap, 0, sizeof(bool) * source->size(), stream);
  thrust::device_ptr<bool> overlap_ptr(d_overlap);

  std::vector<thrust::system::cuda::unique_eager_event> results(targets.size());
  for (int i = 0; i < targets.size(); i++) {
    overlap_count_kernel overlap_kernel(*targets[i], deltas_ptr + i);
    auto first = thrust::make_transform_iterator(thrust::device_ptr<Eigen::Vector3f>(source->points_gpu), overlap_kernel);
    auto last = thrust::make_transform_iterator(thrust::device_ptr<Eigen::Vector3f>(source->points_gpu) + source->size(), overlap_kernel);

    if (i == 0) {
      auto result = thrust::async::transform(
        thrust::cuda::par.on(stream),
        thrust::make_zip_iterator(thrust::make_tuple(overlap_ptr, first)),
        thrust::make_zip_iterator(thrust::make_tuple(overlap_ptr + source->size(), last)),
        overlap_ptr,
        bool_or_kernel());

      results[i] = std::move(result);
    } else {
      auto result = thrust::async::transform(
        thrust::cuda::par.after(results[i - 1]),
        thrust::make_zip_iterator(thrust::make_tuple(overlap_ptr, first)),
        thrust::make_zip_iterator(thrust::make_tuple(overlap_ptr + source->size(), last)),
        overlap_ptr,
        bool_or_kernel());

      results[i] = std::move(result);
    }
  }

  auto result = thrust::async::reduce(
    thrust::cuda::par.after(results.back()),
    thrust::make_transform_iterator(overlap_ptr, cast_kernel<int>()),
    thrust::make_transform_iterator(overlap_ptr + source->size(), cast_kernel<int>()),
    0);
  int num_inliers = result.get();

  cudaFreeAsync(d_deltas, stream);
  cudaFreeAsync(d_overlap, stream);

  return static_cast<double>(num_inliers) / source->size();
}

double overlap_gpu(
  const std::vector<Frame::ConstPtr>& targets,
  const Frame::ConstPtr& source,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas_,
  CUstream_st* stream) {
  std::vector<GaussianVoxelMap::ConstPtr> target_voxelmaps(targets.size());
  std::transform(targets.begin(), targets.end(), target_voxelmaps.begin(), [](const Frame::ConstPtr& frame) { return frame->voxels_gpu; });
  return overlap_gpu(target_voxelmaps, source, deltas_, stream);
}

}  // namespace gtsam_ext