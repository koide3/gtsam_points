// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam/linear/HessianFactor.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>

namespace gtsam_points {

/**
 * @brief This factor adds a constant diagonal matrix to the Hessian matrix.
 *        It causes a damping effect (like lambda in LM optimization) and helps to fix the gauge freedom.
 */
class LinearDampingFactor : public gtsam::LinearContainerFactor {
public:
  using shared_ptr = boost::shared_ptr<LinearDampingFactor>;

  LinearDampingFactor(gtsam::Key key, int dim, double mu) : gtsam::LinearContainerFactor(gtsam::HessianFactor(key, mu * gtsam::Matrix::Identity(dim, dim), gtsam::Vector::Zero(dim), 0.0)) {}

  LinearDampingFactor(gtsam::Key key, const gtsam::Vector& diag) : gtsam::LinearContainerFactor(gtsam::HessianFactor(key, diag.asDiagonal(), gtsam::Vector::Zero(diag.size()), 0.0)) {}

  LinearDampingFactor() {}

  virtual ~LinearDampingFactor() override {}

private:
  /** Serialization function */
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    namespace bs = ::boost::serialization;
    ar& boost::serialization::make_nvp("LinearContainerFactor", boost::serialization::base_object<gtsam::LinearContainerFactor>(*this));
  }
};

}