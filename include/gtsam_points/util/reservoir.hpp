// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <array>
#include <cmath>
#include <random>

namespace gtsam_points {

/// @brief Reservoir sampling for a fixed number of samples k
/// @tparam Type  Type of samples
/// @tparam k     Number of samples to keep
template <typename Type, int k>
struct Reservoir {
public:
  Reservoir() : total_num_samples(0) {}

  size_t size() const { return std::min<size_t>(total_num_samples, k); }

  const Type& front() const { return samples.front(); }
  const Type& back() const { return samples.back(); }
  const Type& operator[](size_t idx) const { return samples[idx]; }

  template <typename RNG>
  void push(const Type& sample, RNG& rng) {
    total_num_samples++;
    if (total_num_samples <= k) {
      samples[total_num_samples - 1] = sample;
    } else {
      std::uniform_int_distribution<size_t> dist(0, total_num_samples - 1);
      const size_t idx = dist(rng);
      if (idx < k) {
        samples[idx] = sample;
      }
    }
  }

public:
  size_t total_num_samples;
  std::array<Type, k> samples;
};

/// @brief Weighted reservoir sampling for a fixed number of samples k
/// @tparam Type  Type of samples
/// @tparam k     Number of samples to keep
template <typename Type, int k>
struct WeightedReservoir {
public:
  WeightedReservoir() : total_num_samples(0), min_key_idx(0) {}

  size_t size() const { return std::min<size_t>(total_num_samples, k); }

  const Type& front() const { return samples.front(); }
  const Type& back() const { return samples.back(); }
  const Type& operator[](size_t idx) const { return samples[idx]; }

  /// @brief Add a weighted sample to the reservoir
  /// @param weight Weight of the sample (must be positive)
  /// @param sample The sample to add
  /// @param rng Random number generator
  template <typename RNG>
  void push(double weight, const Type& sample, RNG& rng) {
    total_num_samples++;
    // Generate key based on weight: key = u^(1/weight) where u ~ Uniform(0,1)
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double u = dist(rng);
    const double key = std::pow(u, 1.0 / weight);

    if (total_num_samples <= k) {
      // Fill the reservoir until we have k samples
      const size_t idx = total_num_samples - 1;
      samples[idx] = sample;
      keys[idx] = key;
      // Update minimum key index
      if (total_num_samples == 1 || key < keys[min_key_idx]) {
        min_key_idx = idx;
      }
    } else {
      // Replace the sample with minimum key if new key is larger
      if (key > keys[min_key_idx]) {
        samples[min_key_idx] = sample;
        keys[min_key_idx] = key;
        // Find new minimum key index
        updateMinKey();
      }
    }
  }

private:
  void updateMinKey() {
    min_key_idx = 0;
    for (size_t i = 1; i < k; i++) {
      if (keys[i] < keys[min_key_idx]) {
        min_key_idx = i;
      }
    }
  }

public:
  size_t total_num_samples;
  size_t min_key_idx;
  std::array<Type, k> samples;
  std::array<double, k> keys;
};

}  // namespace gtsam_points
