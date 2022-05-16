#include <gtsam_ext/optimizers/incremental_fixed_lag_smoother_with_fallback.hpp>

#include <chrono>
#include <unordered_set>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>

namespace gtsam_ext {

IncrementalFixedLagSmootherExtWithFallback::IncrementalFixedLagSmootherExtWithFallback(double smootherLag, const ISAM2Params& parameters)
: IncrementalFixedLagSmootherExt(smootherLag, parameters) {
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

  const double smoother_lag = smoother->smootherLag();
  const double time_horizon = current_stamp - smoother_lag;

  Result result;

  try {
    result = smoother->update(newFactors, newTheta, timestamps, factorsToRemove);
  } catch (...) {
    fallback_smoother();
    result = smoother->update();
  }

  update_fallback_state();

  return result;
}

gtsam::Values IncrementalFixedLagSmootherExtWithFallback::calculateEstimate() const {
  try {
    return smoother->calculateEstimate();
  } catch (...) {
    fallback_smoother();
    return smoother->calculateEstimate();
  }
}

const gtsam::Value& IncrementalFixedLagSmootherExtWithFallback::calculateEstimate(gtsam::Key key) const {
  try {
    const auto& value = smoother->calculateEstimate(key);
    auto found = values.find(key);
    if(found != values.end()) {
      found->value = value;
    }

    return value;
  } catch (...) {
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
    if(!factors_to_eliminate.count(factor.get())) {
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
  std::cout << "falling back!!" << std::endl;
  std::cout << "smoother_lag:" << smoother->smootherLag() << std::endl;

  double min_stamp = std::numeric_limits<double>::max();
  for (const auto& stamp : stamps) {
    min_stamp = std::min<double>(min_stamp, stamp.second);
  }

  const double fixation_duration = 0.5;
  const double fixation_time = min_stamp + fixation_duration;

  gtsam::NonlinearFactorGraph new_factors = factors;
  for (const auto& value : values) {
    const auto found = stamps.find(value.key);
    if (found == stamps.end() || found->second > fixation_time) {
      continue;
    }

    const gtsam::Symbol symbol(value.key);
    if(symbol.chr() != 'x') {
      continue;
    }

    const int dim = value.value.dim();
    gtsam::Matrix G = 1e6 * gtsam::Matrix::Identity(dim, dim);
    gtsam::Vector g = gtsam::Vector::Zero(dim);

    const gtsam::HessianFactor factor(value.key, G, g, 0.0);
    new_factors.emplace_shared<gtsam::LinearContainerFactor>(factor, values);
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
          std::cout << "- " << gtsam::Symbol(key) << std::endl;
        }
      }
    }
  }

  smoother.reset(new IncrementalFixedLagSmootherExt(smoother->smootherLag(), smoother->params()));
  smoother->update(new_factors, values, new_stamps);
}

}  // namespace gtsam_ext