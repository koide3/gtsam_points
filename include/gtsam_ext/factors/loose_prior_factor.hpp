// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam_ext {

// This factor "loosely" fixes a variable at the current estimate point
// This can be used to fix the gauge freedom (there should be a better way though...)
// Note: Consider using LinearDampingFactor instead of this
template <class VALUE>
class LoosePriorFactor : public gtsam::NonlinearFactor {
public:
  using T = VALUE;
  using shared_ptr = boost::shared_ptr<LoosePriorFactor<T>>;

  LoosePriorFactor() {}
  ~LoosePriorFactor() override {}

  LoosePriorFactor(gtsam::Key key, const T& prior, const gtsam::SharedNoiseModel& model = nullptr)
  : gtsam::NonlinearFactor(gtsam::KeyVector{key}),
    noise_model_(model),
    prior_(new gtsam::PriorFactor<T>(key, prior, model)) {}

  virtual size_t dim() const override { return prior_->dim(); }

  virtual double error(const gtsam::Values& values) const override { return prior_->error(values); }

  virtual boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& values) const override {
    T current_value = values.at<T>(keys()[0]);
    if (!prior_ || prior_->error(values) > 1e-3) {
      prior_.reset(new gtsam::PriorFactor<T>(keys()[0], current_value, noise_model_));
    }

    return prior_->linearize(values);
  }

private:
  /** Serialization function */
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    namespace bs = ::boost::serialization;
    ar& boost::serialization::make_nvp("NonlinearFactor", boost::serialization::base_object<gtsam::NonlinearFactor>(*this));
    ar& BOOST_SERIALIZATION_NVP(noise_model_);
  }

private:
  gtsam::SharedNoiseModel noise_model_;
  mutable boost::shared_ptr<gtsam::PriorFactor<T>> prior_;
};

}  // namespace gtsam