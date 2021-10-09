#pragma once

#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam_ext {

class BundleAdjustmentFactorBase : public gtsam::NonlinearFactor {
public:
  using shared_ptr = boost::shared_ptr<BundleAdjustmentFactorBase>;

  virtual ~BundleAdjustmentFactorBase() {}

  virtual int num_points() const = 0;
  virtual void add(const gtsam::Point3& pt, const gtsam::Key& key) = 0;
};

}