// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam_points/util/gtsam_migration.hpp>
#include <gtsam_points/optimizers/isam2_ext.hpp>

namespace gtsam_points {

class ISAM2ExtDummy : public gtsam_points::ISAM2Ext {
public:
  explicit ISAM2ExtDummy(const ISAM2Params& params) {}
  virtual ~ISAM2ExtDummy() {}

  virtual ISAM2ResultExt update(
    const NonlinearFactorGraph& newFactors = NonlinearFactorGraph(),
    const Values& newTheta = Values(),
    const FactorIndices& removeFactorIndices = FactorIndices(),
    const optional<FastMap<Key, int> >& constrainedKeys = {},
    const optional<FastList<Key> >& noRelinKeys = {},
    const optional<FastList<Key> >& extraReelimKeys = {},
    bool force_relinearize = false) override {
    //
    nonlinearFactors_.add(newFactors);
    theta_.insert(newTheta);

    delta_ = theta_.zeroVectors();

    ISAM2ResultExt result;
    return result;
  }
};
}  // namespace gtsam_points