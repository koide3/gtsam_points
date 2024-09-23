// SPDX-License-Identifier: MIT
// Copyright (c) 2024  Kenji Koide (k.koide@aist.go.jp)
#include <gtsam_points/util/parallelism.hpp>

#include <iostream>
#include <gtsam_points/config.hpp>

namespace gtsam_points {

ParallelismBackend default_parallelism = ParallelismBackend::OMP;

void set_tbb_as_default() {
  default_parallelism = ParallelismBackend::TBB;
}

void set_omp_as_default() {
  default_parallelism = ParallelismBackend::OMP;
}

void set_default_parallelism(ParallelismBackend parallelism) {
#ifndef _OPENMP
  if (parallelism == ParallelismBackend::OMP) {
    throw std::runtime_error("OpenMP is not available");
  }
#endif

#ifndef GTSAM_POINTS_USE_TBB
  if (parallelism == ParallelismBackend::TBB) {
    std::cerr << "warning: Intel TBB is not available" << std::endl;
    return;
  }
#endif

  default_parallelism = parallelism;
}

bool is_tbb_default() {
  return default_parallelism == ParallelismBackend::TBB;
}

bool is_omp_default() {
  return default_parallelism == ParallelismBackend::OMP;
}

ParallelismBackend get_default_parallelism() {
  return default_parallelism;
}

}  // namespace gtsam_points
