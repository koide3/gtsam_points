#include <gtsam_ext/types/frame.hpp>

#include <thrust/device_ptr.h>
#include <thrust/async/reduce.h>
#include <thrust/async/transform.h>

#include <gtsam_ext/types/voxelized_frame.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_gpu.hpp>

#include <gtsam_ext/cuda/kernels/vector3_hash.cuh>

namespace gtsam_ext {

// point coord -> voxel coord conversion
struct overlap_count_kernel {
public:
  overlap_count_kernel(const GaussianVoxelMapGPU& voxelmap, const thrust::device_ptr<const Eigen::Isometry3f>& delta_ptr)
  : voxelmap_info_ptr(voxelmap.voxelmap_info_ptr.data()),
    buckets_ptr(voxelmap.buckets.data()),
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

double Frame::overlap_gpu(const std::shared_ptr<VoxelizedFrame>& target, const Eigen::Isometry3f* delta_gpu) const {
  if (!points_gpu || !covs_gpu || !target->voxels_gpu) {
    std::cerr << "error:  GPU (points/covs/target_voxel) has not been created!!" << std::endl;
    abort();
  }

  thrust::device_vector<bool> overlap(num_points);
  auto trans_result = thrust::async::transform(
    thrust::device_ptr<Eigen::Vector3f>(points_gpu),
    thrust::device_ptr<Eigen::Vector3f>(points_gpu) + num_points,
    overlap.begin(),
    overlap_count_kernel(*target->voxels_gpu, thrust::device_ptr<const Eigen::Isometry3f>(delta_gpu)));
  auto reduce_result = thrust::async::reduce(
    thrust::cuda::par.after(trans_result),
    thrust::make_transform_iterator(overlap.begin(), cast_kernel<int>()),
    thrust::make_transform_iterator(overlap.end(), cast_kernel<int>()),
    0);

  int num_inliers = reduce_result.get();

  return static_cast<double>(num_inliers) / num_points;
}

double Frame::overlap_gpu(const std::shared_ptr<VoxelizedFrame>& target, const Eigen::Isometry3d& delta) const {
  if (!points_gpu || !covs_gpu || !target->voxels_gpu) {
    std::cerr << "error:  GPU (points/covs/target_voxel) has not been created!!" << std::endl;
    abort();
  }

  thrust::host_vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>> deltas(1);
  deltas[0] = delta.cast<float>();
  thrust::device_vector<Eigen::Isometry3f> delta_ptr = deltas;

  return overlap_gpu(target, thrust::raw_pointer_cast(delta_ptr.data()));
}

double Frame::overlap_gpu(const std::vector<std::shared_ptr<VoxelizedFrame>>& targets, const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas_)
  const {
  if (!points_gpu || !covs_gpu || std::find_if(targets.begin(), targets.end(), [](const auto& target) { return target == nullptr; }) != targets.end()) {
    std::cerr << "error:  GPU (points/covs/target_voxel) has not been created!!" << std::endl;
    abort();
  }

  thrust::host_vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>> deltas(deltas_.size());
  thrust::transform(deltas_.begin(), deltas_.end(), deltas.begin(), [](const Eigen::Isometry3d& delta) { return delta.cast<float>(); });
  thrust::device_vector<Eigen::Isometry3f> deltas_ptr = deltas;

  thrust::device_vector<bool> overlap(num_points, false);
  std::vector<thrust::system::cuda::unique_eager_event> results(targets.size());

  for (int i = 0; i < targets.size(); i++) {
    overlap_count_kernel overlap_kernel(*targets[i]->voxels_gpu, deltas_ptr.data() + i);
    auto first = thrust::make_transform_iterator(thrust::device_ptr<Eigen::Vector3f>(points_gpu), overlap_kernel);
    auto last = thrust::make_transform_iterator(thrust::device_ptr<Eigen::Vector3f>(points_gpu) + num_points, overlap_kernel);

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

  return static_cast<double>(num_inliers) / num_points;
}

}  // namespace gtsam_ext
