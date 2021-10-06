#pragma once

#include <memory>
#include <unordered_map>
#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam_ext {

struct BALMFeature;

class EVMFactorBase : public gtsam::NonlinearFactor {
public:
  using shared_ptr = boost::shared_ptr<EVMFactorBase>;

  EVMFactorBase();
  virtual ~EVMFactorBase() override;

  virtual size_t dim() const override { return 6; }

  int num_points() const { return points.size(); }
  void add(const gtsam::Point3& pt, const gtsam::Key& key);

  template <int k>
  double calc_eigenvalue(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& transed_points, Eigen::MatrixXd* H = nullptr, Eigen::MatrixXd* J = nullptr)
    const;

  Eigen::MatrixXd calc_pose_derivatives(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& transed_points) const;

protected:
  std::vector<gtsam::Key> keys;
  std::vector<gtsam::Point3, Eigen::aligned_allocator<gtsam::Point3>> points;
  std::unordered_map<gtsam::Key, int> key_index;
};

class PlaneEVMFactor : public EVMFactorBase {
public:
  using shared_ptr = boost::shared_ptr<PlaneEVMFactor>;

  PlaneEVMFactor();
  virtual ~PlaneEVMFactor() override;

  virtual double error(const gtsam::Values& c) const override;
  virtual boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& c) const override;
};

class EdgeEVMFactor : public EVMFactorBase {
public:
  using shared_ptr = boost::shared_ptr<EdgeEVMFactor>;

  EdgeEVMFactor();
  virtual ~EdgeEVMFactor() override;

  virtual double error(const gtsam::Values& c) const override;
  virtual boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& c) const override;
};

}