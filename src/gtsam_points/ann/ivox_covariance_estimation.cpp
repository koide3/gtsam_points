#include <gtsam_points/ann/ivox_covariance_estimation.hpp>

#include <Eigen/Eigen>

namespace gtsam_points {

iVoxCovarianceEstimation::iVoxCovarianceEstimation(
  const double voxel_resolution,
  const double min_points_dist,
  const int lru_thresh,
  const int k_neighbors,
  const int num_threads)
: iVox(voxel_resolution, min_points_dist, lru_thresh),
  k_neighbors(k_neighbors),
  num_threads(num_threads) {}

iVoxCovarianceEstimation::~iVoxCovarianceEstimation() {}

void iVoxCovarianceEstimation::insert(const PointCloud& frame) {
  iVox::insert(frame);

  // Remove covariance voxels that are already removed from iVox
  for (auto cov = covmap.begin(); cov != covmap.end();) {
    if (voxelmap.find(cov->first) == voxelmap.end()) {
      cov = covmap.erase(cov);
    } else {
      cov++;
    }
  }

  // Find covariance voxels corresponding to iVox voxels;
  std::vector<std::pair<Eigen::Vector3i, LinearContainer*>> flat_voxels;
  flat_voxels.reserve(voxelmap.size());
  for (const auto& voxel : voxelmap) {
    flat_voxels.emplace_back(voxel.first, voxel.second.get());
  }

  std::cout << "voxels:" << voxels.size() << std::endl;
  std::cout << "flat_voxels:" << flat_voxels.size() << std::endl;

  std::vector<LinearContainer*> flat_covs(flat_voxels.size());
  covs.resize(voxels.size());

  for (int i = 0; i < flat_voxels.size(); i++) {
    const auto& voxel = flat_voxels[i];
    const auto& coord = voxel.first;
    auto found = covmap.find(coord);

    if (found == covmap.end()) {
      found = covmap.emplace_hint(found, coord, std::make_shared<LinearContainer>(0));
    }

    flat_covs[i] = found->second.get();
    covs[voxel.second->serial_id] = found->second.get();
  }

  // Add new points and estimate their normals and covariances
#pragma omp parallel for num_threads(num_threads) schedule(guided, 4)
  for (int i = 0; i < flat_voxels.size(); i++) {
    const auto& voxel = flat_voxels[i];
    const auto& cov = flat_covs[i];
    cov->normals.reserve(voxel.second->size());
    cov->covs.reserve(voxel.second->size());

    std::vector<size_t> k_indices(k_neighbors);
    std::vector<double> k_sq_dists(k_neighbors);

    for (int j = cov->covs.size(); j < voxel.second->size(); j++) {
      const auto& point = voxel.second->points[j];

      const int num_found = knn_search(point.data(), k_neighbors, k_indices.data(), k_sq_dists.data());

      const auto cov_and_normal = estimate_cov_and_normal(point, num_found, k_indices.data(), k_sq_dists.data());
      cov->covs.emplace_back(cov_and_normal.first);
      cov->normals.emplace_back(cov_and_normal.second);
    }
  }
}

std::pair<Eigen::Matrix4d, Eigen::Vector4d> iVoxCovarianceEstimation::estimate_cov_and_normal(
  const Eigen::Vector4d& point,
  const int num_found,
  const size_t* k_indices,
  const double* k_sq_dists) const {
  if (num_found < 5) {
    Eigen::Matrix4d cov = Eigen::Matrix4d::Identity();
    Eigen::Vector4d normal = Eigen::Vector4d::Random().normalized();

    cov(3, 3) = 0.0;
    normal[3] = 0.0;
    return std::make_pair(cov, normal);
  }

  Eigen::Vector4d sum_points = Eigen::Vector4d::Zero();
  Eigen::Matrix4d sum_covs = Eigen::Matrix4d::Zero();

  for (int i = 0; i < num_found; i++) {
    const auto& pt = frame::point<iVox>(*this, k_indices[i]);
    sum_points += pt;
    sum_covs += pt * pt.transpose();
  }

  const Eigen::Vector4d mean = sum_points / num_found;
  Eigen::Matrix4d cov = (sum_covs - mean * sum_points.transpose()) / num_found;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig;
  eig.computeDirect(cov.block<3, 3>(0, 0));

  cov.setZero();
  cov.block<3, 3>(0, 0) = eig.eigenvectors() * Eigen::Vector3d(1e-3, 1.0, 1.0).asDiagonal() * eig.eigenvectors().inverse();

  Eigen::Vector4d normal = (Eigen::Vector4d() << eig.eigenvectors().col(0), 0.0).finished();
  if (point.dot(normal) > 1.0) {
    normal = -normal;
  }

  return std::make_pair(cov, normal);
}

const Eigen::Vector4d& iVoxCovarianceEstimation::normal(const size_t i) const {
  return covs[voxel_id(i)]->normals[point_id(i)];
}

const Eigen::Matrix4d& iVoxCovarianceEstimation::cov(const size_t i) const {
  return covs[voxel_id(i)]->covs[point_id(i)];
}

std::vector<Eigen::Vector4d> iVoxCovarianceEstimation::voxel_normals() const {
  std::vector<Eigen::Vector4d> normals;
  normals.reserve(covs.size());
  for (const auto& cov : covs) {
    normals.insert(normals.end(), cov->normals.begin(), cov->normals.end());
  }
  return normals;
}

std::vector<Eigen::Matrix4d> iVoxCovarianceEstimation::voxel_covs() const {
  std::vector<Eigen::Matrix4d> covs;
  covs.reserve(this->covs.size());
  for (const auto& cov : this->covs) {
    covs.insert(covs.end(), cov->covs.begin(), cov->covs.end());
  }
  return covs;
}

}  // namespace gtsam_points