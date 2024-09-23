// SPDX-License-Identifier: MIT
// Copyright (c) 2024  Kenji Koide (k.koide@aist.go.jp)
#pragma once

#include <string>

namespace gtsam_points {

/**
 * @brief Parallelism backend
 */
enum class ParallelismBackend {
  OMP,  ///< OpenMP
  TBB   ///< Intel TBB
};

/// @brief Set TBB as the default parallelism backend
void set_tbb_as_default();

/// @brief Set OpenMP as the default parallelism backend
void set_omp_as_default();

/// @brief Set the default parallelism backend
void set_default_parallelism(ParallelismBackend parallelism);

/// @brief Check if the default parallelism backend is TBB
bool is_tbb_default();

/// @brief  Check if the default parallelism backend is OpenMP
bool is_omp_default();

/// @brief Get the default parallelism backend
ParallelismBackend get_default_parallelism();

}  // namespace gtsam_points
