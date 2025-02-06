// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/bundle_adjustment_factor_lsq.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam_points/util/expressions.hpp>

namespace gtsam_points {

struct LsqBundleAdjustmentFactor::FrameDistribution {
public:
  FrameDistribution() {
    update_required = false;
    num_points = 0;
    sum_pts.setZero();
    sum_cross.setZero();
    mean.setZero();
    cov.setZero();
  };

  void add(const gtsam::Point3& pt) {
    update_required = true;
    num_points++;
    sum_pts += pt;
    sum_cross += pt * pt.transpose();
  }

  void finalize() {
    if (!update_required) {
      return;
    }

    mean = sum_pts / num_points;
    cov = (sum_cross - mean * sum_pts.transpose()) / num_points;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    eigenvalues = eig.eigenvalues();
    eigenvectors = eig.eigenvectors();

    update_required = false;
  }

  int num_points;
  Eigen::Vector3d sum_pts;
  Eigen::Matrix3d sum_cross;

  bool update_required;
  Eigen::Vector3d mean;
  Eigen::Matrix3d cov;

  Eigen::Vector3d eigenvalues;
  Eigen::Matrix3d eigenvectors;
};

LsqBundleAdjustmentFactor::LsqBundleAdjustmentFactor() {
  global_num_points = 0;
  global_mean.setZero();
  global_cov.setZero();
}

LsqBundleAdjustmentFactor::~LsqBundleAdjustmentFactor() {}

void LsqBundleAdjustmentFactor::print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const {
  std::cout << s << "LsqBundleAdjustmentFactor";
  std::cout << "(";
  for (const auto& frame : frame_dists) {
    std::cout << keyFormatter(frame.first) << ", ";
  }
  std::cout << ")" << std::endl;
}

void LsqBundleAdjustmentFactor::add(const gtsam::Point3& pt, const gtsam::Key& key) {
  if (std::find(keys_.begin(), keys_.end(), key) == keys_.end()) {
    this->keys_.push_back(key);
  }

  auto found = frame_dists.find(key);
  if (found == frame_dists.end()) {
    std::shared_ptr<FrameDistribution> new_dist(new FrameDistribution);
    found = frame_dists.insert(found, std::make_pair(key, new_dist));
  }

  found->second->add(pt);
  global_num_points++;
}

double LsqBundleAdjustmentFactor::error(const gtsam::Values& c) const {
  if (global_cov.isZero()) {
    update_global_distribution(c);
  }

  double sum_errors = 0.0;

  for (const auto& key : keys_) {
    const auto found = frame_dists.find(key);
    if (found == frame_dists.end()) {
      std::cerr << "error: key " << key << " not found in frame_dists!!" << std::endl;
      abort();
    }

    const gtsam::Pose3 pose = c.at<gtsam::Pose3>(key);
    const Eigen::Matrix3d R_k = pose.rotation().matrix();
    const Eigen::Vector3d t_k = pose.translation();

    const auto& frame_dist = found->second;
    const int n_k = frame_dist->num_points;
    const auto& mean_k = frame_dist->mean;
    const auto& cov_k = frame_dist->cov;
    const auto& lambda = frame_dist->eigenvalues;
    const auto& R_ex = frame_dist->eigenvectors.col(2);
    const auto& R_ey = frame_dist->eigenvectors.col(1);

    sum_errors += n_k * lambda[2] * std::pow((R_k * R_ex).dot(global_normal), 2);
    sum_errors += n_k * lambda[1] * std::pow((R_k * R_ey).dot(global_normal), 2);
    sum_errors += n_k * std::pow((pose * mean_k - global_mean).dot(global_normal), 2);
  }

  return sum_errors;
}

boost::shared_ptr<gtsam::GaussianFactor> LsqBundleAdjustmentFactor::linearize(const gtsam::Values& c) const {
  update_global_distribution(c);

  double sum_errors = 0.0;

  // TODO: Non-diagonal blocks are always zero
  //     : Should use independent Gaussian factors
  std::vector<int> Hs_indices;
  std::vector<gtsam::Matrix> Hs;
  std::vector<gtsam::Vector> bs;
  for (int i = 0; i < keys_.size(); i++) {
    Hs_indices.push_back(Hs.size());
    bs.push_back(gtsam::Vector6::Zero());
    for (int j = i; j < keys_.size(); j++) {
      Hs.push_back(gtsam::Matrix6::Zero());
    }
  }

  for (int i = 0; i < keys_.size(); i++) {
    const auto key = keys_[i];
    const auto found = frame_dists.find(key);
    if (found == frame_dists.end()) {
      std::cerr << "error: key " << key << " not found in frame_dists!!" << std::endl;
      abort();
    }

    const gtsam::Pose3 pose = c.at<gtsam::Pose3>(key);
    const Eigen::Matrix3d R_k = pose.rotation().matrix();
    const Eigen::Vector3d t_k = pose.translation();

    const auto& frame_dist = found->second;
    const int n_k = frame_dist->num_points;
    const auto& mean_k = frame_dist->mean;
    const auto& cov_k = frame_dist->cov;
    const auto& lambda = frame_dist->eigenvalues;
    const auto& R_ex = frame_dist->eigenvectors.col(2);
    const auto& R_ey = frame_dist->eigenvectors.col(1);

    double e1 = (R_k * R_ex).dot(global_normal);
    double e2 = (R_k * R_ey).dot(global_normal);
    double e3 = global_normal.transpose() * (R_k * mean_k + t_k - global_mean);

    sum_errors += n_k * lambda[2] * e1 * e1 + n_k * lambda[1] * e2 * e2 + n_k * e3 * e3;

    auto& H_k = Hs[Hs_indices[i]];
    auto& b_k = bs[i];

    gtsam::Matrix13 H0 = global_normal.transpose() * R_k * -gtsam::SO3::Hat(R_ex);
    H_k.block<3, 3>(0, 0) += H0.transpose() * n_k * lambda[2] * H0;
    b_k.head<3>() -= H0.transpose() * n_k * lambda[2] * e1;

    gtsam::Matrix13 H1 = global_normal.transpose() * R_k * -gtsam::SO3::Hat(R_ey);
    H_k.block<3, 3>(0, 0) += H1.transpose() * n_k * lambda[1] * H1;
    b_k.head<3>() -= H1.transpose() * n_k * lambda[1] * e2;

    gtsam::Matrix36 H2_;
    H2_.block<3, 3>(0, 0) = R_k * -gtsam::SO3::Hat(mean_k);
    H2_.block<3, 3>(0, 3) = R_k * gtsam::Matrix3::Identity();

    gtsam::Matrix16 H2 = global_normal.transpose() * H2_;
    H_k += H2.transpose() * n_k * H2;
    b_k -= H2.transpose() * n_k * e3;
  }

  return gtsam::make_shared<gtsam::HessianFactor>(keys_, Hs, bs, sum_errors);
}

void LsqBundleAdjustmentFactor::update_global_distribution(const gtsam::Values& values) const {
  global_mean.setZero();
  global_cov.setZero();

  for (const auto& frame_dist : frame_dists) {
    frame_dist.second->finalize();

    const gtsam::Pose3 pose_k = values.at<gtsam::Pose3>(frame_dist.first);

    const int n_k = frame_dist.second->num_points;
    const Eigen::Vector3d mean_k = pose_k * frame_dist.second->mean;
    global_mean += n_k / static_cast<double>(global_num_points) * mean_k;
  }

  for (const auto& frame_dist : frame_dists) {
    const gtsam::Pose3 pose_k = values.at<gtsam::Pose3>(frame_dist.first);
    const Eigen::Matrix3d R_k = pose_k.rotation().matrix();

    const int n_k = frame_dist.second->num_points;
    const Eigen::Vector3d mean_k = pose_k * frame_dist.second->mean;
    const Eigen::Matrix3d cov_k = R_k * frame_dist.second->cov * R_k.transpose();
    const Eigen::Vector3d diff_k = mean_k - global_mean;
    global_cov += n_k / static_cast<double>(global_num_points) * (cov_k + diff_k * diff_k.transpose());
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(global_cov);
  global_normal = eig.eigenvectors().col(0);
}

}  // namespace gtsam_points