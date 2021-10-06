#include <gtsam_ext/factors/bundle_adjustment_factor_evm.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam_ext/factors/balm_feature.hpp>

namespace gtsam_ext {

PlaneEVMFactor::PlaneEVMFactor() {}

PlaneEVMFactor::~PlaneEVMFactor() {}

double PlaneEVMFactor::error(const gtsam::Values& values) const {
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> transed_points(points.size());
  for (int i = 0; i < points.size(); i++) {
    transed_points[i] = values.at<gtsam::Pose3>(keys[i]) * points[i];
  }
  BALMFeature feature(transed_points);

  std::cout << "error:" << feature.eigenvalues.transpose() << std::endl;
  return feature.eigenvalues[0];
}

boost::shared_ptr<gtsam::GaussianFactor> PlaneEVMFactor::linearize(const gtsam::Values& values) const {
  std::cout << ">> linearize" << std::endl;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> transed_points(points.size());
  for (int i = 0; i < points.size(); i++) {
    transed_points[i] = values.at<gtsam::Pose3>(keys[i]) * points[i];
  }
  BALMFeature feature(transed_points);

  Eigen::MatrixXd H(points.size() * 3, points.size() * 3);
  Eigen::MatrixXd J(1, points.size() * 3);

  // TODO: Calculate only the upper triangle
  for(int i=0; i<points.size(); i++) {
    for (int j = 0; j < points.size(); j++) {
      H.block<3, 3>(i * 3, j * 3) = feature.Hij<0>(transed_points[i], transed_points[j], i == j);
    }
  }

  for(int i=0; i<points.size(); i++) {
    J.block<1, 3>(0, i * 3) = feature.Ji<0>(transed_points[i]);
  }

  Eigen::MatrixXd D = Eigen::MatrixXd::Zero(3 * points.size(), 6 * keys_.size());
  for (int i = 0; i < transed_points.size(); i++) {
    Eigen::Matrix<double, 3, 6> Dij;
    Dij.block<3, 3>(0, 0) = -gtsam::SO3::Hat(transed_points[i]);
    Dij.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

    auto found = key_index.find(keys[i]);
    if(found == key_index.end()) {
      std::cerr << "error: key doesn't exist!!" << std::endl;
      abort();
    }

    int index = found->second;
    D.block<3, 6>(3 * i, 6 * index) = Dij;
  }

  Eigen::MatrixXd JD = -J * D;
  Eigen::MatrixXd DHD = D.transpose() * H * D;

  std::vector<gtsam::Key> js;
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

  std::cout << "<< linearize" << std::endl;

  return boost::shared_ptr<gtsam::GaussianFactor>(new gtsam::HessianFactor(keys_, Gs, gs, feature.eigenvalues[0]));
}

void PlaneEVMFactor::add(const gtsam::Point3& pt, const gtsam::Key& key) {
  if (std::find(keys_.begin(), keys_.end(), key) == keys_.end()) {
    key_index[key] = this->keys_.size();
    this->keys_.push_back(key);
  }

  this->keys.push_back(key);
  this->points.push_back(pt);
}
}