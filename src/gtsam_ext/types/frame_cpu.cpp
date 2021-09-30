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

template <typename T, int D>
FrameCPU::FrameCPU(
  const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& points,
  const std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>>& covs) {
  //
  points_storage.resize(points.size(), Eigen::Vector4d(0, 0, 0, 1));
  covs_storage.resize(covs.size(), Eigen::Matrix4d::Zero());

  for (int i = 0; i < points.size(); i++) {
    points_storage[i].head<D>() = points[i].template head<D>().template cast<double>();
    covs_storage[i].block<D, D>(0, 0) = covs[i].template block<D, D>(0, 0).template cast<double>();
  }

  this->num_points = points.size();
  this->points = &points_storage[0];
  this->covs = &covs_storage[0];
}

template FrameCPU::FrameCPU(const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>&);
template FrameCPU::FrameCPU(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>&);
template FrameCPU::FrameCPU(const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>&);
template FrameCPU::FrameCPU(const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>&);

template FrameCPU::FrameCPU(
  const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>&,
  const std::vector<Eigen::Matrix<float, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 3>>>&);
template FrameCPU::FrameCPU(
  const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>&,
  const std::vector<Eigen::Matrix<float, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 4>>>&);
template FrameCPU::FrameCPU(
  const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>&,
  const std::vector<Eigen::Matrix<double, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 3>>>&);
template FrameCPU::FrameCPU(
  const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>&,
  const std::vector<Eigen::Matrix<double, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>>&);

FrameCPU::~FrameCPU() {}

}  // namespace gtsam_ext
