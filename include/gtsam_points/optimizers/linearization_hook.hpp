// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

namespace gtsam_points {

class NonlinearFactorSet {
public:
  virtual ~NonlinearFactorSet() {}

  virtual int size() const = 0;
  virtual void clear() = 0;
  virtual void clear_counts() = 0;

  virtual int linearization_count() const = 0;
  virtual int evaluation_count() const = 0;

  virtual bool add(boost::shared_ptr<gtsam::NonlinearFactor> factor) = 0;
  virtual void add(const gtsam::NonlinearFactorGraph& factors) = 0;

  virtual void linearize(const gtsam::Values& values) = 0;
  virtual void error(const gtsam::Values& values) = 0;

  virtual std::vector<gtsam::GaussianFactor::shared_ptr> calc_linear_factors(const gtsam::Values& linearization_point) = 0;
};

class LinearizationHook {
public:
  LinearizationHook();
  ~LinearizationHook();

  int size() const;
  void clear();
  void clear_counts();

  int linearization_count() const;
  int evaluation_count() const;

  bool add(boost::shared_ptr<gtsam::NonlinearFactor> factor);
  void add(const gtsam::NonlinearFactorGraph& factors) ;

  void linearize(const gtsam::Values& values);
  void error(const gtsam::Values& values);

  std::vector<gtsam::GaussianFactor::shared_ptr> calc_linear_factors(const gtsam::Values& linearization_point);

  static void register_hook(const std::function<std::shared_ptr<NonlinearFactorSet>()>& hook);

private:
  static std::vector<std::function<std::shared_ptr<NonlinearFactorSet>()>> hook_constructors;

  std::vector<std::shared_ptr<NonlinearFactorSet>> hooks;
};

}  // namespace gtsam_points