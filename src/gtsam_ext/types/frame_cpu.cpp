#include <gtsam_ext/types/frame_cpu.hpp>

namespace gtsam_ext {

template <typename T, int D>
FrameCPU::FrameCPU(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& points) {
  points_storage.resize(points.size(), Eigen::Vector4d(0, 0, 0, 1));
  for (int i = 0; i < points.size(); i++) {
    points_storage[i].head<D>() = points[i].template head<D>().template cast<double>();
  }

  this->num_points = points.size();
  this->points = &points_storage[0];
}

FrameCPU::~FrameCPU() {}

template <typename T>
void FrameCPU::add_times(const std::vector<T>& times) {
  assert(times.size() == size());
  times_storage.resize(times.size());
  std::transform(times.begin(), times.end(), times_storage.begin(), [](const auto& t) { return static_cast<double>(t); });
  this->times = times_storage.data();
}

template <typename T, int D>
void FrameCPU::add_normals(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& normals) {
  assert(normals.size() == size());
  normals_storage.resize(normals.size(), Eigen::Vector4d::Zero());
  for (int i = 0; i < normals.size(); i++) {
    normals_storage[i].head<D>() = normals[i].template head<D>().template cast<double>();
  }
  this->normals = normals_storage.data();
}

template <typename T, int D>
void FrameCPU::add_covs(const std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>>& covs) {
  assert(covs.size() == size());
  covs_storage.resize(covs.size(), Eigen::Matrix4d::Zero());
  for (int i = 0; i < covs.size(); i++) {
    covs_storage[i].block<D, D>(0, 0) = covs[i].template block<D, D>(0, 0).template cast<double>();
  }
  this->covs = covs_storage.data();
}

template FrameCPU::FrameCPU(const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>&);
template FrameCPU::FrameCPU(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>&);
template FrameCPU::FrameCPU(const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>&);
template FrameCPU::FrameCPU(const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>&);

template void FrameCPU::add_times(const std::vector<float>& times);
template void FrameCPU::add_times(const std::vector<double>& times);
template void FrameCPU::add_normals(const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>& normals);
template void FrameCPU::add_normals(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>& normals);
template void FrameCPU::add_normals(const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>& normals);
template void FrameCPU::add_normals(const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>& normals);
template void FrameCPU::add_covs(const std::vector<Eigen::Matrix<float, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 3>>>& covs);
template void FrameCPU::add_covs(const std::vector<Eigen::Matrix<float, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 4>>>& covs);
template void FrameCPU::add_covs(const std::vector<Eigen::Matrix<double, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 3>>>& covs);
template void FrameCPU::add_covs(const std::vector<Eigen::Matrix<double, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>>& covs);

}  // namespace gtsam_ext
