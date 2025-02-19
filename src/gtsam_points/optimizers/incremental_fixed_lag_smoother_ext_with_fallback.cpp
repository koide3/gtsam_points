// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/optimizers/incremental_fixed_lag_smoother_with_fallback.hpp>

#include <chrono>
#include <unordered_set>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam_points/factors/linear_damping_factor.hpp>

namespace gtsam_points {

IncrementalFixedLagSmootherExtWithFallback::IncrementalFixedLagSmootherExtWithFallback(double smootherLag, const ISAM2Params& parameters)
: IncrementalFixedLagSmootherExt(smootherLag, parameters),
  fix_variable_types({{'x', 0}}) {
  fallback_happend = false;
  current_stamp = 0.0;
  smoother.reset(new IncrementalFixedLagSmootherExt(smootherLag, parameters));
}

IncrementalFixedLagSmootherExtWithFallback::~IncrementalFixedLagSmootherExtWithFallback() {}

IncrementalFixedLagSmootherExtWithFallback::Result IncrementalFixedLagSmootherExtWithFallback::update(
  const gtsam::NonlinearFactorGraph& newFactors,
  const gtsam::Values& newTheta,  //
  const KeyTimestampMap& timestamps,
  const gtsam::FactorIndices& factorsToRemove) {
  //
  fallback_happend = false;
  factors.add(newFactors);
  for (const auto& factor : newFactors) {
    for (const auto key : factor->keys()) {
      factor_map[key].push_back(factor);
    }
  }

  values.insert(newTheta);
  for (auto& stamp : timestamps) {
    stamps[stamp.first] = stamp.second;
    current_stamp = std::max(current_stamp, stamp.second);
  }

  Result result;

  try {
    result = smoother->update(newFactors, newTheta, timestamps, factorsToRemove);
  } catch (std::exception& e) {
    std::cerr << "warning: an exception was caught in fixed-lag smoother update!!" << std::endl;
    std::cerr << "       : " << e.what() << std::endl;

    fallback_smoother();
    result = smoother->update();
  }

  update_fallback_state();

  return result;
}

gtsam::Values IncrementalFixedLagSmootherExtWithFallback::calculateEstimate() const {
  try {
    return smoother->calculateEstimate();
  } catch (std::exception& e) {
    std::cerr << "warning: an exception was caught in fixed-lag smoother calculateEstimate!!" << std::endl;
    std::cerr << "       : " << e.what() << std::endl;

    fallback_smoother();
    return smoother->calculateEstimate();
  }
}

const gtsam::Value& IncrementalFixedLagSmootherExtWithFallback::calculateEstimate(gtsam::Key key) const {
  try {
    const auto& value = smoother->calculateEstimate(key);
    auto found = values.find(key);
    if (found != values.end()) {
      found->value = value;
    }

    return value;
  } catch (std::exception& e) {
    std::cerr << "warning: an exception was caught in fixed-lag smoother calculateEstimate!!" << std::endl;
    std::cerr << "       : " << e.what() << std::endl;

    fallback_smoother();
    return smoother->calculateEstimate(key);
  }
}

void IncrementalFixedLagSmootherExtWithFallback::update_fallback_state() {
  if (current_stamp < 1e-6) {
    return;
  }

  const double smoother_lag = smoother->smootherLag();
  const double time_horizon = current_stamp - smoother_lag;

  // Find variables to be eliminated
  std::unordered_set<gtsam::Key> variables_to_eliminate;
  for (const auto& value : values) {
    const auto found = stamps.find(value.key);
    if (found != stamps.end()) {
      if (found->second < time_horizon) {
        variables_to_eliminate.insert(value.key);
      }
    }
  }

  std::unordered_set<gtsam::NonlinearFactor*> factors_to_eliminate;
  for (const auto& key : variables_to_eliminate) {
    const auto& related_factors = factor_map[key];
    for (const auto& factor : related_factors) {
      factors_to_eliminate.insert(factor.get());
    }

    values.erase(key);
  }

  for (auto itr = stamps.begin(); itr != stamps.end();) {
    if (variables_to_eliminate.count(itr->first)) {
      itr = stamps.erase(itr);
    } else {
      itr++;
    }
  }

  gtsam::NonlinearFactorGraph next_factors;
  next_factors.reserve(factors.size());
  for (const auto& factor : factors) {
    if (!factors_to_eliminate.count(factor.get())) {
      next_factors.add(factor);
    }
  }

  factors = std::move(next_factors);

  for (auto itr = factor_map.begin(); itr != factor_map.end();) {
    if (variables_to_eliminate.count(itr->first)) {
      itr = factor_map.erase(itr);
    } else {
      itr++;
    }
  }
}

void IncrementalFixedLagSmootherExtWithFallback::fallback_smoother() const {
  fallback_happend = true;
  std::cout << "falling back!!" << std::endl;
  std::cout << "smoother_lag:" << smoother->smootherLag() << std::endl;

  // Find basic value statistics
  std::unordered_map<char, double> min_stamps;
  std::unordered_map<char, std::pair<size_t, size_t>> minmax_ids;
  for (const auto& value : values) {
    const auto found = stamps.find(value.key);
    if (found == stamps.end()) {
      std::cerr << "warning: corresponding stamp is not found for " << gtsam::Symbol(value.key) << std::endl;
      continue;
    }

    const double stamp = found->second;
    const gtsam::Symbol symbol(value.key);

    auto min_stamp = min_stamps.find(symbol.chr());
    if (min_stamp == min_stamps.end()) {
      min_stamps.emplace_hint(min_stamp, symbol.chr(), stamp);
    } else {
      min_stamp->second = std::min(min_stamp->second, stamp);
    }

    auto minmax_id = minmax_ids.find(symbol.chr());
    if (minmax_id == minmax_ids.end()) {
      minmax_ids.emplace_hint(minmax_id, symbol.chr(), std::make_pair(symbol.index(), symbol.index()));
    } else {
      minmax_id->second.first = std::min(minmax_id->second.first, symbol.index());
      minmax_id->second.second = std::max(minmax_id->second.second, symbol.index());
    }
  }

  std::cout << boost::format("current_stamp:%.6f") % current_stamp << std::endl;
  const double fixation_duration = 1.0;
  for (const auto& min_stamp : min_stamps) {
    const char chr = min_stamp.first;
    const auto minmax_id = minmax_ids[chr];

    std::cout << boost::format("- symbol=%c min_id=%d max_id=%d min_stamp=%.6f fixation=%.6f") % chr % minmax_id.first % minmax_id.second %
                   min_stamp.second % (min_stamp.second + fixation_duration)
              << std::endl;
  }

  // connectivity check
  std::unordered_map<gtsam::Key, std::vector<gtsam::NonlinearFactor*>> factormap;
  for (const auto& factor : factors) {
    for (const auto key : factor->keys()) {
      factormap[key].push_back(factor.get());
    }
  }

  std::unordered_set<gtsam::Key> traversed_keys;
  std::vector<gtsam::NonlinearFactor*> traverse_queue;
  for (const auto& minmax_id : minmax_ids) {
    const auto chr = minmax_id.first;
    const auto max_id = minmax_id.second.second;
    const gtsam::Symbol symbol(chr, max_id);
    const auto found = factormap.find(symbol);
    if (found != factormap.end()) {
      traversed_keys.insert(symbol);
      traverse_queue.insert(traverse_queue.end(), found->second.begin(), found->second.end());
    } else {
      std::cerr << "symbol not found:" << symbol << std::endl;
    }
  }

  while (!traverse_queue.empty()) {
    const auto factor = traverse_queue.back();
    traverse_queue.pop_back();

    for (const auto key : factor->keys()) {
      if (traversed_keys.count(key)) {
        continue;
      }

      traversed_keys.insert(key);
      const auto found = factormap.find(key);
      if (found != factormap.end()) {
        traverse_queue.insert(traverse_queue.end(), found->second.begin(), found->second.end());
      }
    }
  }

  std::unordered_set<gtsam::Key> keys_to_remove;
  for (const auto& value : values) {
    if (!traversed_keys.count(value.key)) {
      std::cerr << "unreached key found:" << gtsam::Symbol(value.key) << std::endl;
      keys_to_remove.insert(value.key);
    }
  }

  for (const auto& key : keys_to_remove) {
    values.erase(key);
    stamps.erase(key);
  }

  // Copy only valid factors
  gtsam::NonlinearFactorGraph new_factors;
  new_factors.reserve(factors.size());
  for (const auto& factor : factors) {
    if (std::all_of(factor->keys().begin(), factor->keys().end(), [&](const auto key) { return values.exists(key); })) {
      new_factors.add(factor);
    } else {
      std::cerr << "remove factor:";
      for (const auto key : factor->keys()) {
        std::cerr << " " << gtsam::Symbol(key);
      }
      std::cerr << std::endl;
    }
  }
  this->factors = new_factors;

  // Create fixation factors
  for (const auto& value : values) {
    const gtsam::Symbol symbol(value.key);
    const double fixation_time = min_stamps[symbol.chr()] + fixation_duration;

    const auto stamp = stamps.find(value.key);
    if (stamp == stamps.end() || stamp->second > fixation_time) {
      continue;
    }

    const int dim = value.value.dim();
    new_factors.emplace_shared<gtsam_points::LinearDampingFactor>(value.key, dim, 1e6);

    for (const auto var_type : fix_variable_types) {
      if (symbol.chr() == var_type.first) {
        std::cout << "fixing " << symbol << std::endl;
        switch (var_type.second) {
          case 0:
            new_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
              value.key,
              value.value.cast<gtsam::Pose3>(),
              gtsam::noiseModel::Isotropic::Precision(6, 1e6));
            break;
          case 1:
            new_factors.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
              value.key,
              value.value.cast<gtsam::Vector3>(),
              gtsam::noiseModel::Isotropic::Precision(3, 1e6));
            break;
          default:
            std::cerr << "error: unknown variable type!! (chr=" << static_cast<int>(var_type.first) << " type=" << var_type.second << ")"
                      << std::endl;
            break;
        }
      }
    }
  }

  const double smoother_lag = smoother->smootherLag();
  const double time_horizon = current_stamp - smoother_lag;

  gtsam::FixedLagSmootherKeyTimestampMap new_stamps;
  for (auto& stamp : stamps) {
    new_stamps[stamp.first] = std::max(time_horizon + 0.5, stamp.second);
  }

  std::cout << "factors:" << factors.size() << " fixed:" << new_factors.size() - factors.size() << " values:" << values.size()
            << " stamps:" << stamps.size() << std::endl;

  for (const auto& factor : new_factors) {
    for (const auto key : factor->keys()) {
      if (!values.exists(key)) {
        std::cout << "illegal factor found!!" << std::endl;
        for (const auto key : factor->keys()) {
          std::cout << "- " << gtsam::Symbol(key) << " exists:" << values.exists(key) << std::endl;
        }
      }
    }
  }

  for (const auto& stamp : stamps) {
    if (!values.exists(stamp.first)) {
      std::cout << "illegal stamp found!! " << gtsam::Symbol(stamp.first) << std::endl;
    }
  }

  smoother.reset(new IncrementalFixedLagSmootherExt(smoother->smootherLag(), smoother->params()));
  smoother->update(new_factors, values, new_stamps);
}

}  // namespace gtsam_points