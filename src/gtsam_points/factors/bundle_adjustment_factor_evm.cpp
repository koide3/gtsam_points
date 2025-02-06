// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/bundle_adjustment_factor_evm.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam_points/factors/balm_feature.hpp>

namespace gtsam_points {

EVMBundleAdjustmentFactorBase::EVMBundleAdjustmentFactorBase() : error_scale(1.0) {}

EVMBundleAdjustmentFactorBase::~EVMBundleAdjustmentFactorBase() {}

void EVMBundleAdjustmentFactorBase::print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const {
  std::cout << s << "EVMBundleAdjustmentFactorBase";
  std::cout << "(";
  for (int i = 0; i < keys.size(); i++) {
    if (i) {
      std::cout << ", ";
    }
    std::cout << keyFormatter(keys[i]);
  }
  std::cout << ")" << std::endl;
  std::cout << "|points|=" << points.size() << ", error_scale=" << error_scale << std::endl;
}

void EVMBundleAdjustmentFactorBase::set_scale(double scale) {
  error_scale = scale;
}

void EVMBundleAdjustmentFactorBase::add(const gtsam::Point3& pt, const gtsam::Key& key) {
  if (std::find(keys_.begin(), keys_.end(), key) == keys_.end()) {
    key_index[key] = this->keys_.size();
    this->keys_.push_back(key);
  }

  this->keys.push_back(key);
  this->points.push_back(pt);
}

/**
 * @brief Calculate k-th eigenvalue and its derivatives
 *        k = 0 is the smallest eigenvalue
 */
template <int k>
double EVMBundleAdjustmentFactorBase::calc_eigenvalue(const std::vector<Eigen::Vector3d>& transed_points, Eigen::MatrixXd* H, Eigen::MatrixXd* J)
  const {
  BALMFeature feature(transed_points);
  if (H == nullptr || J == nullptr) {
    return feature.eigenvalues[k];
  }

  *H = Eigen::MatrixXd::Zero(points.size() * 3, points.size() * 3);
  *J = Eigen::MatrixXd(1, points.size() * 3);

  for (int i = 0; i < points.size(); i++) {
    for (int j = i; j < points.size(); j++) {
      H->block<3, 3>(i * 3, j * 3) = feature.Hij<k>(transed_points[i], transed_points[j], i == j);
    }
  }

  for (int i = 0; i < points.size(); i++) {
    J->block<1, 3>(0, i * 3) = feature.Ji<k>(transed_points[i]);
  }

  return feature.eigenvalues[k];
}

/**
 * @brief Calculate dp / dT
 * @ref   Eqs. (9) - (12)
 */
Eigen::MatrixXd EVMBundleAdjustmentFactorBase::calc_pose_derivatives(const std::vector<Eigen::Vector3d>& transed_points) const {
  Eigen::MatrixXd D = Eigen::MatrixXd::Zero(3 * points.size(), 6 * keys_.size());
  for (int i = 0; i < transed_points.size(); i++) {
    Eigen::Matrix<double, 3, 6> Dij;
    Dij.block<3, 3>(0, 0) = -gtsam::SO3::Hat(transed_points[i]);
    Dij.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

    auto found = key_index.find(keys[i]);
    if (found == key_index.end()) {
      std::cerr << "error: key doesn't exist!!" << std::endl;
      abort();
    }

    int index = found->second;
    D.block<3, 6>(3 * i, 6 * index) = Dij;
  }

  return D;
}

/**
 * @brief Compose a Hessian factor from derivatives
 */
gtsam::GaussianFactor::shared_ptr EVMBundleAdjustmentFactorBase::compose_factor(const Eigen::MatrixXd& H, const Eigen::MatrixXd& J, double error)
  const {
  std::vector<gtsam::Matrix> Gs;
  std::vector<gtsam::Vector> gs;

  for (int i = 0; i < keys_.size(); i++) {
    for (int j = i; j < keys_.size(); j++) {
      Gs.push_back(H.block<6, 6>(i * 6, j * 6));
    }
  }
  for (int i = 0; i < keys_.size(); i++) {
    gs.push_back(J.block<1, 6>(0, i * 6));
  }

  return boost::shared_ptr<gtsam::HessianFactor>(new gtsam::HessianFactor(keys_, Gs, gs, error));
}

/**
 * @brief PlaneEVMFactor
 */
PlaneEVMFactor::PlaneEVMFactor() : EVMBundleAdjustmentFactorBase() {}

PlaneEVMFactor::~PlaneEVMFactor() {}

void PlaneEVMFactor::print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const {
  std::cout << s << "PlaneEVMFactor";
  std::cout << "(";
  for (int i = 0; i < keys.size(); i++) {
    if (i) {
      std::cout << ", ";
    }
    std::cout << keyFormatter(keys[i]);
  }
  std::cout << ")" << std::endl;
  std::cout << "|points|=" << points.size() << ", error_scale=" << error_scale << std::endl;
}

double PlaneEVMFactor::error(const gtsam::Values& values) const {
  std::vector<Eigen::Vector3d> transed_points(points.size());
  for (int i = 0; i < points.size(); i++) {
    transed_points[i] = values.at<gtsam::Pose3>(keys[i]) * points[i];
  }
  return error_scale * calc_eigenvalue<0>(transed_points);
}

boost::shared_ptr<gtsam::GaussianFactor> PlaneEVMFactor::linearize(const gtsam::Values& values) const {
  std::vector<Eigen::Vector3d> transed_points(points.size());
  for (int i = 0; i < points.size(); i++) {
    transed_points[i] = values.at<gtsam::Pose3>(keys[i]) * points[i];
  }

  Eigen::MatrixXd H, J;
  double lambda_0 = calc_eigenvalue<0>(transed_points, &H, &J);

  Eigen::MatrixXd D = calc_pose_derivatives(transed_points);

  // Eq. (13)
  Eigen::MatrixXd JD = -J * D;
  Eigen::MatrixXd DHD = D.transpose() * H.selfadjointView<Eigen::Upper>() * D;

  return compose_factor(error_scale * DHD, error_scale * JD, error_scale * lambda_0);
}

/**
 * @brief EdgeEVMFactor
 */
EdgeEVMFactor::EdgeEVMFactor() : EVMBundleAdjustmentFactorBase() {}

EdgeEVMFactor::~EdgeEVMFactor() {}

void EdgeEVMFactor::print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const {
  std::cout << s << "EdgeEVMFactor";
  std::cout << "(";
  for (int i = 0; i < keys.size(); i++) {
    if (i) {
      std::cout << ", ";
    }
    std::cout << keyFormatter(keys[i]);
  }
  std::cout << ")" << std::endl;
  std::cout << "|points|=" << points.size() << ", error_scale=" << error_scale << std::endl;
}

double EdgeEVMFactor::error(const gtsam::Values& values) const {
  std::vector<Eigen::Vector3d> transed_points(points.size());
  for (int i = 0; i < points.size(); i++) {
    transed_points[i] = values.at<gtsam::Pose3>(keys[i]) * points[i];
  }
  return calc_eigenvalue<0>(transed_points) + calc_eigenvalue<1>(transed_points);
}

boost::shared_ptr<gtsam::GaussianFactor> EdgeEVMFactor::linearize(const gtsam::Values& values) const {
  std::vector<Eigen::Vector3d> transed_points(points.size());
  for (int i = 0; i < points.size(); i++) {
    transed_points[i] = values.at<gtsam::Pose3>(keys[i]) * points[i];
  }

  Eigen::MatrixXd H0, J0, H1, J1;
  double lambda_0 = calc_eigenvalue<0>(transed_points, &H0, &J0);
  double lambda_1 = calc_eigenvalue<1>(transed_points, &H1, &J1);

  Eigen::MatrixXd D = calc_pose_derivatives(transed_points);

  Eigen::MatrixXd H = H0 + H1;
  Eigen::MatrixXd J = J0 + J1;

  // Eq. (13)
  Eigen::MatrixXd JD = -J * D;
  Eigen::MatrixXd DHD = D.transpose() * H.selfadjointView<Eigen::Upper>() * D;

  return compose_factor(DHD, JD, lambda_0 + lambda_1);
}
}  // namespace gtsam_points