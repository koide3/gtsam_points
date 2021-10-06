#pragma once

#include <memory>
#include <unordered_map>
#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam_ext {

struct BALMFeature;

class PlaneEVMFactor : public gtsam::NonlinearFactor {
public:
  using shared_ptr = boost::shared_ptr<PlaneEVMFactor>;

  PlaneEVMFactor();
  virtual ~PlaneEVMFactor() override;

  virtual size_t dim() const override { return 6; }

  virtual double error(const gtsam::Values& c) const override;

  virtual boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& c) const override;

  void add(const gtsam::Point3& pt, const gtsam::Key& key);

  int num_points() const { return points.size(); }


private:
  std::vector<gtsam::Key> keys;
  std::vector<gtsam::Point3, Eigen::aligned_allocator<gtsam::Point3>> points;
  std::unordered_map<gtsam::Key, int> key_index;
};
}