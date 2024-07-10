// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam_points {

/**
 * @brief Base class of range-based bundle adjustment factors
 */
class BundleAdjustmentFactorBase : public gtsam::NonlinearFactor {
public:
  using shared_ptr = boost::shared_ptr<BundleAdjustmentFactorBase>;

  virtual ~BundleAdjustmentFactorBase() {}

  /**
   * @brief Assign a point to the factor
   * @param pt  Point
   * @param key Key of the frame corresponding to the point
   */
  virtual void add(const gtsam::Point3& pt, const gtsam::Key& key) = 0;

  /**
   * @brief Number of points assigned to the factor
   * @return Number of points
   */
  virtual int num_points() const = 0;
  
  /**
   * @brief Set a constant error scale to boost the weight of the constraint
   */
  virtual void set_scale(double scale) {}
};
}