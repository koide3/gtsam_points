#include <gtsam_ext/types/voxelized_frame_gpu.hpp>

#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_gpu.hpp>

namespace gtsam_ext {

// note: (points|covs)_storage must be allocated before calling this
void VoxelizedFrameGPU::init(double voxel_resolution) {
  // CPU data
  num_points = points_storage.size();
  points = &points_storage[0];
  covs = &covs_storage[0];

  voxels_storage.reset(new GaussianVoxelMapCPU(voxel_resolution));
  voxels_storage->create_voxelmap(*this);
  voxels = voxels_storage.get();

  // GPU data
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> host_points(num_points);
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> host_covs(num_points);
  std::transform(points_storage.begin(), points_storage.end(), host_points.begin(), [](const Eigen::Vector4d& pt) { return pt.head<3>().cast<float>(); });
  std::transform(covs_storage.begin(), covs_storage.end(), host_covs.begin(), [](const Eigen::Matrix4d& cov) { return cov.block<3, 3>(0, 0).cast<float>(); });

  times_gpu_storage.reset(new FloatsGPU);
  points_gpu_storage.reset(new PointsGPU);
  normals_gpu_storage.reset(new PointsGPU);
  covs_gpu_storage.reset(new MatricesGPU);

  points_gpu_storage->resize(num_points);
  covs_gpu_storage->resize(num_points);
  cudaMemcpy(thrust::raw_pointer_cast(points_gpu_storage->data()), host_points.data(), sizeof(Eigen::Vector3f) * num_points, cudaMemcpyHostToDevice);
  cudaMemcpy(thrust::raw_pointer_cast(covs_gpu_storage->data()), host_covs.data(), sizeof(Eigen::Matrix3f) * num_points, cudaMemcpyHostToDevice);

  points_gpu = thrust::raw_pointer_cast(points_gpu_storage->data());
  covs_gpu = thrust::raw_pointer_cast(covs_gpu_storage->data());

  voxels_gpu_storage.reset(new GaussianVoxelMapGPU(voxel_resolution));
  voxels_gpu_storage->create_voxelmap(*this);
  voxels_gpu = voxels_gpu_storage.get();
}

VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& points_,
  const std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>>& covs_) {
  // CPU data
  points_storage.resize(points_.size());
  covs_storage.resize(covs_.size());
  for (int i = 0; i < points_.size(); i++) {
    points_storage[i] << points_[i].cast<double>(), 1.0f;
    covs_storage[i].setZero();
    covs_storage[i].block<3, 3>(0, 0) = covs_[i].cast<double>();
  }

  init(voxel_resolution);
}

VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& points_,
  const std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& covs_) {
  // CPU data
  points_storage.resize(points_.size());
  covs_storage.resize(covs_.size());
  for (int i = 0; i < points_.size(); i++) {
    points_storage[i] << points_[i], 1.0;
    covs_storage[i].setZero();
    covs_storage[i].block<3, 3>(0, 0) = covs_[i];
  }

  init(voxel_resolution);
}

VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points_,
  const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs_) {
  // CPU data
  points_storage = points_;
  covs_storage = covs_;

  init(voxel_resolution);
}

VoxelizedFrameGPU::VoxelizedFrameGPU(
  double voxel_resolution,
  const PointsGPU& points,
  const MatricesGPU& covs,
  bool allocate_cpu) {
  // GPU data
  times_gpu_storage.reset(new FloatsGPU);
  points_gpu_storage.reset(new PointsGPU);
  normals_gpu_storage.reset(new PointsGPU);
  covs_gpu_storage.reset(new MatricesGPU);

  this->num_points = points.size();
  points_gpu_storage->resize(num_points);
  covs_gpu_storage->resize(num_points);
  cudaMemcpy(thrust::raw_pointer_cast(points_gpu_storage->data()), thrust::raw_pointer_cast(points.data()), sizeof(Eigen::Vector3f) * points.size(), cudaMemcpyDeviceToDevice);
  cudaMemcpy(thrust::raw_pointer_cast(covs_gpu_storage->data()), thrust::raw_pointer_cast(covs.data()), sizeof(Eigen::Matrix3f) * covs.size(), cudaMemcpyDeviceToDevice);

  this->points_gpu = thrust::raw_pointer_cast(points_gpu_storage->data());
  this->covs_gpu = thrust::raw_pointer_cast(covs_gpu_storage->data());

  voxels_gpu_storage.reset(new GaussianVoxelMapGPU(voxel_resolution));
  voxels_gpu_storage->create_voxelmap(*this);
  voxels_gpu = voxels_gpu_storage.get();

  if(allocate_cpu) {
    thrust::host_vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> host_points(num_points);
    thrust::host_vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> host_covs(num_points);
    cudaMemcpy(thrust::raw_pointer_cast(host_points.data()), points_gpu, sizeof(Eigen::Vector3f) * num_points, cudaMemcpyDeviceToHost);
    cudaMemcpy(thrust::raw_pointer_cast(host_covs.data()), covs_gpu, sizeof(Eigen::Matrix3f) * num_points, cudaMemcpyDeviceToHost);

    points_storage.resize(num_points, Eigen::Vector4d(0.0, 0.0, 0.0, 1.0));
    covs_storage.resize(num_points, Eigen::Matrix4d::Zero());
    for (int i = 0; i < num_points; i++) {
      points_storage[i].head<3>() = host_points[i].cast<double>();
      covs_storage[i].block<3, 3>(0, 0) = host_covs[i].cast<double>();
    }

    this->points = &points_storage[0];
    this->covs = &covs_storage[0];

    voxels_storage.reset(new GaussianVoxelMapCPU(voxel_resolution));
    voxels_storage->create_voxelmap(*this);
    voxels = voxels_storage.get();
  }
}

VoxelizedFrameGPU::~VoxelizedFrameGPU() {}

template <typename T>
void VoxelizedFrameGPU::add_times(const std::vector<T>& times) {
  assert(times.size() == size());
  times_storage.resize(times.size());
  thrust::transform(times.begin(), times.end(), times_storage.begin(), [](const auto& v) { return static_cast<double>(v); });
  this->times = times_storage.data();

  add_times_gpu(times);
}

template <typename T>
void VoxelizedFrameGPU::add_times_gpu(const std::vector<T>& times) {
  assert(times.size() == size());

  thrust::host_vector<float> times_h(times.size());
  std::transform(times.begin(), times.end(), times_h.begin(), [](const auto& v) { return static_cast<float>(v); });

  times_gpu_storage->resize(times.size());
  cudaMemcpy(thrust::raw_pointer_cast(times_gpu_storage->data()), times_h.data(), sizeof(float) * times.size(), cudaMemcpyHostToDevice);
  this->times_gpu = thrust::raw_pointer_cast(times_gpu_storage->data());
}

template <typename T, int D>
void VoxelizedFrameGPU::add_normals(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& normals) {
  assert(normals.size() == size());
  normals_storage.resize(normals.size(), Eigen::Vector4d(0, 0, 0, 1));
  for (int i = 0; i < normals.size(); i++) {
    normals_storage[i].template head<D>() = normals[i].template cast<double>();
  }
  this->normals = normals_storage.data();

  add_normals_gpu(normals);
}

template <typename T, int D>
void VoxelizedFrameGPU::add_normals_gpu(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& normals) {
  assert(normals.size() == size());

  thrust::host_vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> normals_h(normals.size());
  for (int i = 0; i < normals.size(); i++) {
    normals_h[i] = normals[i].template head<3>().template cast<float>();
  }

  normals_gpu_storage->resize(normals.size());
  cudaMemcpy(thrust::raw_pointer_cast(normals_gpu_storage->data()), normals_h.data(), sizeof(Eigen::Vector3f) * normals.size(), cudaMemcpyHostToDevice);
  this->normals_gpu = thrust::raw_pointer_cast(normals_gpu_storage->data());
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

template void VoxelizedFrameGPU::add_times(const std::vector<float>&);
template void VoxelizedFrameGPU::add_times(const std::vector<double>&);

template void VoxelizedFrameGPU::add_normals(const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>&);
template void VoxelizedFrameGPU::add_normals(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>&);
template void VoxelizedFrameGPU::add_normals(const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>&);
template void VoxelizedFrameGPU::add_normals(const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>&);

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
}

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