// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/optimizers/linearization_hook.hpp>

#include <numeric>

namespace gtsam_points {

std::vector<std::function<std::shared_ptr<NonlinearFactorSet>()>> LinearizationHook::hook_constructors;

void LinearizationHook::register_hook(const std::function<std::shared_ptr<NonlinearFactorSet>()>& hook) {
  hook_constructors.emplace_back(hook);
}

LinearizationHook::LinearizationHook() {
  hooks.resize(hook_constructors.size());
  for (int i = 0; i < hook_constructors.size(); i++) {
    hooks[i] = hook_constructors[i]();
  }
}

LinearizationHook::~LinearizationHook() {}

int LinearizationHook::size() const {
  int count = 0;
  for(const auto& hook: hooks) {
    count += hook->size();
  }
  return count;
}

void LinearizationHook::clear() {
  for (auto& hook : hooks) {
    hook->clear();
  }
}

void LinearizationHook::clear_counts() {
  for (auto& hook : hooks) {
    hook->clear_counts();
  }
}

int LinearizationHook::linearization_count() const {
  int count = 0;
  for (const auto& hook : hooks) {
    count += hook->linearization_count();
  }
  return count;
}

int LinearizationHook::evaluation_count() const {
  int count = 0;
  for (const auto& hook : hooks) {
    count += hook->evaluation_count();
  }
  return count;
}

bool LinearizationHook::add(boost::shared_ptr<gtsam::NonlinearFactor> factor) {
  bool inserted = false;
  for (auto& hook : hooks) {
    inserted |= hook->add(factor);
  }
  return inserted;
}

void LinearizationHook::add(const gtsam::NonlinearFactorGraph& factors) {
  for (auto& hook : hooks) {
    hook->add(factors);
  }
}

void LinearizationHook::linearize(const gtsam::Values& values) {
  for (auto& hook : hooks) {
    hook->linearize(values);
  }
}

void LinearizationHook::error(const gtsam::Values& values) {
  for (auto& hook : hooks) {
    hook->error(values);
  }
}

std::vector<gtsam::GaussianFactor::shared_ptr> LinearizationHook::calc_linear_factors(const gtsam::Values& linearization_point) {
  std::vector<gtsam::GaussianFactor::shared_ptr> linear_factors;
  for (auto& hook : hooks) {
    auto factors = hook->calc_linear_factors(linearization_point);
    linear_factors.insert(linear_factors.end(), factors.begin(), factors.end());
  }
  return linear_factors;
}

}  // namespace gtsam_points
