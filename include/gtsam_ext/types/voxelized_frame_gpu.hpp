#pragma once

#include <gtsam_ext/types/voxelized_frame.hpp>

// forward declaration
namespace thrust {

template <typename T>
class device_allocator;

template <typename T, typename Alloc>
class device_vector;

}  // namespace thrust

namespace gtsam_ext {

struct VoxelizedFrameGPU : public VoxelizedFrame {
public:
  using FloatsGPU = thrust::device_vector<float, thrust::device_allocator<float>>;
  using PointsGPU = thrust::device_vector<Eigen::Vector3f, thrust::device_allocator<Eigen::Vector3f>>;
  using MatricesGPU = thrust::device_vector<Eigen::Matrix3f, thrust::device_allocator<Eigen::Matrix3f>>;

  using Ptr = std::shared_ptr<VoxelizedFrameGPU>;
  using ConstPtr = std::shared_ptr<const VoxelizedFrameGPU>;

  template <typename T, int D>
  VoxelizedFrameGPU(double voxel_resolution, const Eigen::Matrix<T, D, 1>* points, const Eigen::Matrix<T, D, D>* covs, int num_points, bool allocate_cpu = true);
  template <typename T, int D>
  VoxelizedFrameGPU(
    double voxel_resolution,
    const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& points,
    const std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>>& covs,
    bool allocate_cpu = true);
  VoxelizedFrameGPU(double voxel_resolution, const Frame& frame, bool allocate_cpu = true);
  VoxelizedFrameGPU(double voxel_resolution, const PointsGPU& points, const MatricesGPU& covs, bool allocate_cpu = false);
  VoxelizedFrameGPU();
  ~VoxelizedFrameGPU();

  void create_voxelmap(double voxel_resolution);

  // add_*_gpu() only adds attributes to GPU storage
  template <typename T>
  void add_times(const T* times, int num_points);
  template <typename T>
  void add_times_gpu(const T* times, int num_points);
  template <typename T>
  void add_times(const std::vector<T>& times);
  template <typename T>
  void add_times_gpu(const std::vector<T>& times);

  template <typename T, int D>
  void add_points(const Eigen::Matrix<T, D, 1>* points, int num_points);
  template <typename T, int D>
  void add_points_gpu(const Eigen::Matrix<T, D, 1>* points, int num_points);
  template <typename T, int D>
  void add_points(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& points);
  template <typename T, int D>
  void add_points_gpu(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& points);

  template <typename T, int D>
  void add_normals(const Eigen::Matrix<T, D, 1>* normals, int num_points);
  template <typename T, int D>
  void add_normals_gpu(const Eigen::Matrix<T, D, 1>* normals, int num_points);
  template <typename T, int D>
  void add_normals(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& normals);
  template <typename T, int D>
  void add_normals_gpu(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& normals);

  template <typename T, int D>
  void add_covs(const Eigen::Matrix<T, D, D>* covs, int num_points);
  template <typename T, int D>
  void add_covs_gpu(const Eigen::Matrix<T, D, D>* covs, int num_points);
  template <typename T, int D>
  void add_covs(const std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>>& covs);
  template <typename T, int D>
  void add_covs_gpu(const std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>>& covs);

  // copy data from GPU to CPU
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> get_points_gpu() const;
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> get_covs_gpu() const;

  std::vector<std::pair<Eigen::Vector3i, int>> get_voxel_buckets_gpu() const;
  std::vector<int> get_voxel_num_points_gpu() const;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> get_voxel_means_gpu() const;
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> get_voxel_covs_gpu() const;

private:
  void init(double voxel_resolution);

public:
  std::shared_ptr<GaussianVoxelMapCPU> voxels_storage;
  std::vector<double> times_storage;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points_storage;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> normals_storage;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> covs_storage;

  std::shared_ptr<GaussianVoxelMapGPU> voxels_gpu_storage;
  std::unique_ptr<FloatsGPU> times_gpu_storage;
  std::unique_ptr<PointsGPU> points_gpu_storage;
  std::unique_ptr<PointsGPU> normals_gpu_storage;
  std::unique_ptr<MatricesGPU> covs_gpu_storage;
};

}  // namespace gtsam_ext
