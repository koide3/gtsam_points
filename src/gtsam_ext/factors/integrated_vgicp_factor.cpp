#include <gtsam_ext/factors/integrated_vgicp_factor.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>

namespace gtsam_ext {

IntegratedVGICPFactor::IntegratedVGICPFactor(gtsam::Key target_key, gtsam::Key source_key, const VoxelizedFrame::ConstPtr& target, const Frame::ConstPtr& source)
: gtsam_ext::IntegratedMatchingCostFactor(target_key, source_key),
  num_threads(1),
  target(target),
  source(source) {
  //
  if (!target->points || !source->points) {
    std::cerr << "error: target or source points has not been allocated!!" << std::endl;
    abort();
  }

  if (!target->voxels) {
    std::cerr << "error: target voxelmap has not been created!!" << std::endl;
    abort();
  }
}

IntegratedVGICPFactor::~IntegratedVGICPFactor() {}

void IntegratedVGICPFactor::update_correspondences(const Eigen::Isometry3d& delta) const {
  correspondences.resize(source->size());
  mahalanobis.resize(source->size());

#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
  for (int i = 0; i < source->size(); i++) {
    Eigen::Vector4d pt = delta * source->points[i];
    Eigen::Vector3i coord = target->voxels->voxel_coord(pt);
    auto voxel = target->voxels->lookup_voxel(coord);

    if (voxel == nullptr) {
      correspondences[i] = nullptr;
      mahalanobis[i].setIdentity();
    } else {
      correspondences[i] = voxel;

      Eigen::Matrix4d RCR = (voxel->cov + delta.matrix() * source->covs[i] * delta.matrix().transpose());
      RCR(3, 3) = 1.0;
      mahalanobis[i] = RCR.inverse();
      mahalanobis[i](3, 3) = 0.0;
    }
  }
}

double IntegratedVGICPFactor::evaluate(
  const Eigen::Isometry3d& delta,
  Eigen::Matrix<double, 6, 6>* H_target,
  Eigen::Matrix<double, 6, 6>* H_source,
  Eigen::Matrix<double, 6, 6>* H_target_source,
  Eigen::Matrix<double, 6, 1>* b_target,
  Eigen::Matrix<double, 6, 1>* b_source) const {
  //
  double sum_errors = 0.0;

  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs_target;
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs_source;
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs_target_source;
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs_target;
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs_source;

  if (H_target && H_source && H_target_source && b_target && b_source) {
    Hs_target.resize(num_threads, Eigen::Matrix<double, 6, 6>::Zero());
    Hs_source.resize(num_threads, Eigen::Matrix<double, 6, 6>::Zero());
    Hs_target_source.resize(num_threads, Eigen::Matrix<double, 6, 6>::Zero());
    bs_target.resize(num_threads, Eigen::Matrix<double, 6, 1>::Zero());
    bs_source.resize(num_threads, Eigen::Matrix<double, 6, 1>::Zero());
  }

#pragma omp parallel for num_threads(num_threads) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < source->size(); i++) {
    const auto& target_voxel = correspondences[i];
    if (target_voxel == nullptr) {
      continue;
    }

    const auto& mean_A = source->points[i];
    const auto& cov_A = source->covs[i];

    const auto& mean_B = target_voxel->mean;
    const auto& cov_B = target_voxel->cov;

    Eigen::Vector4d transed_mean_A = delta * mean_A;
    Eigen::Vector4d error = mean_B - transed_mean_A;

    sum_errors += 0.5 * error.transpose() * mahalanobis[i] * error;

    if (Hs_target.empty()) {
      continue;
    }

    Eigen::Matrix<double, 4, 6> J_target = Eigen::Matrix<double, 4, 6>::Zero();
    J_target.block<3, 3>(0, 0) = -gtsam::SO3::Hat(transed_mean_A.head<3>());
    J_target.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> J_source = Eigen::Matrix<double, 4, 6>::Zero();
    J_source.block<3, 3>(0, 0) = delta.linear() * gtsam::SO3::Hat(mean_A.head<3>());
    J_source.block<3, 3>(0, 3) = -delta.linear();

    int thread_num = 1;
#ifdef _OPENMP
    thread_num = omp_get_thread_num();
#endif

    Hs_target[thread_num] += J_target.transpose() * mahalanobis[i] * J_target;
    Hs_source[thread_num] += J_source.transpose() * mahalanobis[i] * J_source;
    Hs_target_source[thread_num] += J_target.transpose() * mahalanobis[i] * J_source;
    bs_target[thread_num] += J_target.transpose() * mahalanobis[i] * error;
    bs_source[thread_num] += J_source.transpose() * mahalanobis[i] * error;
  }

  if (H_target && H_source && H_target_source && b_target && b_source) {
    H_target->setZero();
    H_source->setZero();
    H_target_source->setZero();
    b_target->setZero();
    b_source->setZero();

    for (int i = 0; i < num_threads; i++) {
      (*H_target) += Hs_target[i];
      (*H_source) += Hs_source[i];
      (*H_target_source) += Hs_target_source[i];
      (*b_target) += bs_target[i];
      (*b_source) += bs_source[i];
    }
  }

  return sum_errors;
}

}  // namespace gtsam_ext
