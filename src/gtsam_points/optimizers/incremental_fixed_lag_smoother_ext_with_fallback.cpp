#include <gtsam_points/optimizers/incremental_fixed_lag_smoother_with_fallback.hpp>

#include <chrono>
#include <unordered_set>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>

namespace gtsam_points {

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
    if(found != values.end()) {
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

  std::unordered_map<char, double> min_stamps;
  std::unordered_map<char, size_t> max_ids;
  for (const auto& value : values) {
    const auto found = stamps.find(value.key);
    if(found == stamps.end()) {
      std::cerr << "warning: corresponding stamp is not found for " << gtsam::Symbol(value.key) << std::endl;
      continue;
    }

    const double stamp = found->second;
    const gtsam::Symbol symbol(value.key);

    auto min_stamp = min_stamps.find(symbol.chr());
    if(min_stamp == min_stamps.end()) {
      min_stamps.emplace_hint(min_stamp, symbol.chr(), stamp);
    } else {
      min_stamp->second = std::min(min_stamp->second, stamp);
    }

    auto max_id = max_ids.find(symbol.chr());
    if(max_id == max_ids.end()) {
      max_ids.emplace_hint(max_id, symbol.chr(), symbol.index());
    } else {
      max_id->second = std::max<size_t>(max_id->second, max_id->second);
    }
  }

  std::cout << boost::format("current_stamp:%.6f") % current_stamp << std::endl;
  const double fixation_duration = 1.0;
  for (const auto& min_stamp : min_stamps) {
    std::cout << boost::format("- symbol=%c min_stamp=%.6f fixation=%.6f") % min_stamp.first % min_stamp.second %
                   (min_stamp.second + fixation_duration)
              << std::endl;
  }

  // connectivity check
  std::unordered_map<gtsam::Key, std::vector<gtsam::NonlinearFactor*>> factormap;
  for (const auto& factor : factors) {
    for(const auto key: factor->keys()) {
      factormap[key].push_back(factor.get());
    }
  }

  std::unordered_set<gtsam::Key> traversed_keys;
  std::vector<gtsam::NonlinearFactor*> traverse_queue;
  for (const auto& max_id : max_ids) {
    const gtsam::Symbol symbol(max_id.first, max_id.second);
    const auto found = factormap.find(symbol);
    if (found != factormap.end()) {
      traversed_keys.insert(symbol);
      traverse_queue.insert(traverse_queue.end(), found->second.begin(), found->second.end());
    }
  }

  while (!traverse_queue.empty()) {
    const auto factor = traverse_queue.back();
    traverse_queue.pop_back();

    for (const auto key : factor->keys()) {
      if(traversed_keys.count(key)) {
        continue;
      }

      traversed_keys.insert(key);
      const auto found = factormap.find(key);
      if(found != factormap.end()) {
        traverse_queue.insert(traverse_queue.end(), found->second.begin(), found->second.end());
      }
    }
  }

  std::unordered_set<gtsam::Key> keys_to_remove;
  for(const auto& value: values) {
    if(!traversed_keys.count(value.key)) {
      std::cerr << "unreached key found:" << gtsam::Symbol(value.key) << std::endl;
      keys_to_remove.insert(value.key);
    }
  }

  for(const auto& key: keys_to_remove) {
    values.erase(key);
    stamps.erase(key);
  }

  // Create fixation factors
  gtsam::NonlinearFactorGraph new_factors = factors;
  for (const auto& value : values) {
    const gtsam::Symbol symbol(value.key);
    const double fixation_time = min_stamps[symbol.chr()] + fixation_duration;

    const auto stamp = stamps.find(value.key);
    if (stamp == stamps.end() || stamp->second > fixation_time) {
      continue;
    }

    const int dim = value.value.dim();
    gtsam::Matrix G = 1e6 * gtsam::Matrix::Identity(dim, dim);
    gtsam::Vector g = gtsam::Vector::Zero(dim);

    const gtsam::HessianFactor factor(value.key, G, g, 0.0);
    new_factors.emplace_shared<gtsam::LinearContainerFactor>(factor, values);

    if (symbol.chr() == 'x') {
      new_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        value.key,
        gtsam::Pose3(value.value.cast<gtsam::Pose3>()),
        gtsam::noiseModel::Isotropic::Precision(6, 1e6));
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

  for(const auto& stamp: stamps) {
    if(!values.exists(stamp.first)) {
      std::cout << "illegal stamp found!! " << gtsam::Symbol(stamp.first) << std::endl;
    }
  }

  smoother.reset(new IncrementalFixedLagSmootherExt(smoother->smootherLag(), smoother->params()));
  smoother->update(new_factors, values, new_stamps);
}

}  // namespace gtsam_points