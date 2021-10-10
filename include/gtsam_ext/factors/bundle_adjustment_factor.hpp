#pragma once

#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam_ext {

class BundleAdjustmentFactorBase : public gtsam::NonlinearFactor {
public:
  using shared_ptr = boost::shared_ptr<BundleAdjustmentFactorBase>;

  virtual ~BundleAdjustmentFactorBase() {}

  // The number of points assigned to this factor (feature)
  virtual int num_points() const = 0;

  // Assign a point to this factor
  virtual void add(const gtsam::Point3& pt, const gtsam::Key& key) = 0;
};

}