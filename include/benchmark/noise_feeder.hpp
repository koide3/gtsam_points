#pragma once

#include <vector>
#include <string>

/// @brief Random noise feeder
/// @details This class generates random noise for the benchmark. It can also read noise from a file for reproducibility.
struct NoiseFeeder {
public:
  /// @brief Default constructor
  /// @details Generates random noise using a normal distribution.
  NoiseFeeder();

  /// @brief Read noise from a text file.
  void read(const std::string& path);

  /// @brief Take the next noise value.
  double operator()() { return noise[(count++) % noise.size()]; }

public:
  int count;                  ///< Generated noise count
  std::vector<double> noise;  ///< Noise values
};