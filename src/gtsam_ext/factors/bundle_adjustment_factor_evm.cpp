#include <gtsam_ext/factors/bundle_adjustment_factor_evm.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam_ext/factors/balm_feature.hpp>

namespace gtsam_ext {

EVMFactorBase::EVMFactorBase() {}

EVMFactorBase::~EVMFactorBase() {}

void EVMFactorBase::add(const gtsam::Point3& pt, const gtsam::Key& key) {
  if (std::find(keys_.begin(), keys_.end(), key) == keys_.end()) {
    key_index[key] = this->keys_.size();
    this->keys_.push_back(key);
  }

  this->keys.push_back(key);
  this->points.push_back(pt);
}

template <int k>
double EVMFactorBase::calc_eigenvalue(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& transed_points, Eigen::MatrixXd* H, Eigen::MatrixXd* J) const {
  BALMFeature feature(transed_points);
  if(H == nullptr || J == nullptr) {
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

Eigen::MatrixXd EVMFactorBase::calc_pose_derivatives(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& transed_points) const {
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

PlaneEVMFactor::PlaneEVMFactor() : EVMFactorBase() {}

PlaneEVMFactor::~PlaneEVMFactor() {}

double PlaneEVMFactor::error(const gtsam::Values& values) const {
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> transed_points(points.size());
  for (int i = 0; i < points.size(); i++) {
    transed_points[i] = values.at<gtsam::Pose3>(keys[i]) * points[i];
  }
  return calc_eigenvalue<0>(transed_points);
}

boost::shared_ptr<gtsam::GaussianFactor> PlaneEVMFactor::linearize(const gtsam::Values& values) const {
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> transed_points(points.size());
  for (int i = 0; i < points.size(); i++) {
    transed_points[i] = values.at<gtsam::Pose3>(keys[i]) * points[i];
  }

  Eigen::MatrixXd H, J;
  double error = calc_eigenvalue<0>(transed_points, &H, &J);

  Eigen::MatrixXd D = calc_pose_derivatives(transed_points);

  Eigen::MatrixXd JD = -J * D;
  Eigen::MatrixXd DHD = D.transpose() * H.selfadjointView<Eigen::Upper>() * D;

  std::vector<gtsam::Matrix> Gs;
  std::vector<gtsam::Vector> gs;

  for(int i=0; i<keys_.size(); i++) {
    for(int j=i; j<keys_.size(); j++) {
      Gs.push_back(DHD.block<6, 6>(i * 6, j * 6));
    }
  }
  for(int i=0; i<keys_.size(); i++) {
    gs.push_back(JD.block<1, 6>(0, i * 6));
  }

  return boost::shared_ptr<gtsam::GaussianFactor>(new gtsam::HessianFactor(keys_, Gs, gs, error));
}

EdgeEVMFactor::EdgeEVMFactor() : EVMFactorBase() {}

EdgeEVMFactor::~EdgeEVMFactor() {}

double EdgeEVMFactor::error(const gtsam::Values& values) const {
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> transed_points(points.size());
  for (int i = 0; i < points.size(); i++) {
    transed_points[i] = values.at<gtsam::Pose3>(keys[i]) * points[i];
  }
  return calc_eigenvalue<0>(transed_points) + calc_eigenvalue<1>(transed_points);
}

boost::shared_ptr<gtsam::GaussianFactor> EdgeEVMFactor::linearize(const gtsam::Values& values) const {
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> transed_points(points.size());
  for (int i = 0; i < points.size(); i++) {
    transed_points[i] = values.at<gtsam::Pose3>(keys[i]) * points[i];
  }

  Eigen::MatrixXd H0, J0, H1, J1;
  double lambda_0 = calc_eigenvalue<0>(transed_points, &H0, &J0);
  double lambda_1 = calc_eigenvalue<1>(transed_points, &H1, &J1);

  Eigen::MatrixXd D = calc_pose_derivatives(transed_points);

  Eigen::MatrixXd H = H0 + H1;
  Eigen::MatrixXd J = J0 + J1;

  Eigen::MatrixXd JD = -J * D;
  Eigen::MatrixXd DHD = D.transpose() * H.selfadjointView<Eigen::Upper>() * D;

  std::vector<gtsam::Matrix> Gs;
  std::vector<gtsam::Vector> gs;

  for (int i = 0; i < keys_.size(); i++) {
    for (int j = i; j < keys_.size(); j++) {
      Gs.push_back(DHD.block<6, 6>(i * 6, j * 6));
    }
  }
  for (int i = 0; i < keys_.size(); i++) {
    gs.push_back(JD.block<1, 6>(0, i * 6));
  }

  return boost::shared_ptr<gtsam::GaussianFactor>(new gtsam::HessianFactor(keys_, Gs, gs, lambda_0 + lambda_1));
}
}