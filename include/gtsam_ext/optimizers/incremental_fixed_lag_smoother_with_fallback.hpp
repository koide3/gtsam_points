// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)
#pragma once

#include <gtsam_ext/optimizers/incremental_fixed_lag_smoother_ext.hpp>

namespace gtsam_ext {

class IncrementalFixedLagSmootherExtWithFallback : public IncrementalFixedLagSmootherExt {
public:
  IncrementalFixedLagSmootherExtWithFallback(double smootherLag, const ISAM2Params& parameters);
  ~IncrementalFixedLagSmootherExtWithFallback() override;

  Result update(
    const gtsam::NonlinearFactorGraph& newFactors = gtsam::NonlinearFactorGraph(),
    const gtsam::Values& newTheta = gtsam::Values(),  //
    const KeyTimestampMap& timestamps = KeyTimestampMap(),
    const gtsam::FactorIndices& factorsToRemove = gtsam::FactorIndices()) override;

  gtsam::Values calculateEstimate() const override;

  const gtsam::Value& calculateEstimate(gtsam::Key key) const;

  template <class VALUE>
  VALUE calculateEstimate(Key key) const {
    try {
      const auto& value = smoother->calculateEstimate(key);
      auto found = values.find(key);
      if (found != values.end()) {
        found->value = value;
      }

      return value.cast<VALUE>();
    } catch (std::exception& e) {
      std::cerr << "warning: an exception was caught in fixed-lag smoother update!!" << std::endl;
      std::cerr << "       : " << e.what() << std::endl;

      fallback_smoother();

      const auto& value = smoother->calculateEstimate(key);
      auto found = values.find(key);
      if (found != values.end()) {
        found->value = value;
      }

      return value.cast<VALUE>();
    }
  }

  Matrix marginalCovariance(Key key) const { return smoother->marginalCovariance(key); }

private:
  void update_fallback_state();
  void fallback_smoother() const;

private:
  double current_stamp;
  mutable gtsam::Values values;
  gtsam::NonlinearFactorGraph factors;
  std::unordered_map<gtsam::Key, std::vector<gtsam::NonlinearFactor::shared_ptr>> factor_map;
  gtsam::FixedLagSmootherKeyTimestampMap stamps;

  mutable std::unique_ptr<IncrementalFixedLagSmootherExt> smoother;
};
}  // namespace gtsam_ext