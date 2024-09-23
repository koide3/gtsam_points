// SPDX-License-Identifier: MIT
// Copyright (c) 2024  Kenji Koide (k.koide@aist.go.jp)
#pragma once

#include <Eigen/Core>
#include <gtsam_points/config.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#endif

namespace gtsam_points {

template <typename Transform>
double scan_matching_reduce_omp(
  const Transform f,
  int num_points,
  int num_threads,
  Eigen::Matrix<double, 6, 6>* H_target,
  Eigen::Matrix<double, 6, 6>* H_source,
  Eigen::Matrix<double, 6, 6>* H_target_source,
  Eigen::Matrix<double, 6, 1>* b_target,
  Eigen::Matrix<double, 6, 1>* b_source) {
  double sum_errors = 0.0;

  const int num_Hs = H_target ? num_threads : 0;
  std::vector<Eigen::Matrix<double, 6, 6>> Hs_target(num_Hs, Eigen::Matrix<double, 6, 6>::Zero());
  std::vector<Eigen::Matrix<double, 6, 6>> Hs_source(num_Hs, Eigen::Matrix<double, 6, 6>::Zero());
  std::vector<Eigen::Matrix<double, 6, 6>> Hs_target_source(num_Hs, Eigen::Matrix<double, 6, 6>::Zero());
  std::vector<Eigen::Matrix<double, 6, 1>> bs_target(num_Hs, Eigen::Matrix<double, 6, 1>::Zero());
  std::vector<Eigen::Matrix<double, 6, 1>> bs_source(num_Hs, Eigen::Matrix<double, 6, 1>::Zero());

#pragma omp parallel for num_threads(num_threads) schedule(guided, 8) reduction(+ : sum_errors)
  for (int i = 0; i < num_points; i++) {
    int thread_num = 0;
#ifdef _OPENMP
    thread_num = omp_get_thread_num();
#endif

    double error = 0.0;
    if (Hs_target.empty()) {
      error = f(i, nullptr, nullptr, nullptr, nullptr, nullptr);
    } else {
      error = f(i, &Hs_target[thread_num], &Hs_source[thread_num], &Hs_target_source[thread_num], &bs_target[thread_num], &bs_source[thread_num]);
    }

    sum_errors += error;
  }

  if (H_target) {
    *H_target = Hs_target[0];
    *H_source = Hs_source[0];
    *H_target_source = Hs_target_source[0];
    *b_target = bs_target[0];
    *b_source = bs_source[0];

    for (int i = 1; i < num_threads; i++) {
      *H_target += Hs_target[i];
      *H_source += Hs_source[i];
      *H_target_source += Hs_target_source[i];
      *b_target += bs_target[i];
      *b_source += bs_source[i];
    }
  }

  return sum_errors;
}

#ifdef GTSAM_POINTS_USE_TBB

template <typename Transform>
struct ScanMatchingReductionTBBError {
public:
  ScanMatchingReductionTBBError(const Transform f) : f(f), sum_errors(0.0) {}
  ScanMatchingReductionTBBError(const ScanMatchingReductionTBBError& other, tbb::split) : f(other.f), sum_errors(0.0) {}

  void operator()(const tbb::blocked_range<int>& range) {
    double local_sum_errors = sum_errors;
    for (int i = range.begin(); i != range.end(); i++) {
      local_sum_errors += f(i, nullptr, nullptr, nullptr, nullptr, nullptr);
    }
    sum_errors = local_sum_errors;
  }

  void join(const ScanMatchingReductionTBBError& other) { sum_errors += other.sum_errors; }

public:
  const Transform f;
  double sum_errors;
};

template <typename Transform>
struct ScanMatchingReductionTBBLinearize {
public:
  ScanMatchingReductionTBBLinearize(const Transform f)
  : f(f),
    sum_errors(0.0),
    H_target(Eigen::Matrix<double, 6, 6>::Zero()),
    H_source(Eigen::Matrix<double, 6, 6>::Zero()),
    H_target_source(Eigen::Matrix<double, 6, 6>::Zero()),
    b_target(Eigen::Matrix<double, 6, 1>::Zero()),
    b_source(Eigen::Matrix<double, 6, 1>::Zero()) {}

  ScanMatchingReductionTBBLinearize(const ScanMatchingReductionTBBLinearize& other, tbb::split)
  : f(other.f),
    sum_errors(0.0),
    H_target(Eigen::Matrix<double, 6, 6>::Zero()),
    H_source(Eigen::Matrix<double, 6, 6>::Zero()),
    H_target_source(Eigen::Matrix<double, 6, 6>::Zero()),
    b_target(Eigen::Matrix<double, 6, 1>::Zero()),
    b_source(Eigen::Matrix<double, 6, 1>::Zero()) {}

  void operator()(const tbb::blocked_range<int>& range) {
    double local_sum_errors = sum_errors;
    Eigen::Matrix<double, 6, 6> local_H_target = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 6> local_H_source = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 6> local_H_target_source = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> local_b_target = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 1> local_b_source = Eigen::Matrix<double, 6, 1>::Zero();

    for (int i = range.begin(); i != range.end(); i++) {
      local_sum_errors += f(i, &local_H_target, &local_H_source, &local_H_target_source, &local_b_target, &local_b_source);
    }

    sum_errors = local_sum_errors;
    H_target += local_H_target;
    H_source += local_H_source;
    H_target_source += local_H_target_source;
    b_target += local_b_target;
    b_source += local_b_source;
  }

  void join(const ScanMatchingReductionTBBLinearize& other) {
    sum_errors += other.sum_errors;
    H_target += other.H_target;
    H_source += other.H_source;
    H_target_source += other.H_target_source;
    b_target += other.b_target;
    b_source += other.b_source;
  }

public:
  const Transform f;
  double sum_errors;
  Eigen::Matrix<double, 6, 6> H_target;
  Eigen::Matrix<double, 6, 6> H_source;
  Eigen::Matrix<double, 6, 6> H_target_source;
  Eigen::Matrix<double, 6, 1> b_target;
  Eigen::Matrix<double, 6, 1> b_source;
};

template <typename Transform>
double scan_matching_reduce_tbb(
  const Transform f,
  int num_points,
  Eigen::Matrix<double, 6, 6>* H_target,
  Eigen::Matrix<double, 6, 6>* H_source,
  Eigen::Matrix<double, 6, 6>* H_target_source,
  Eigen::Matrix<double, 6, 1>* b_target,
  Eigen::Matrix<double, 6, 1>* b_source) {
  if (H_target) {
    ScanMatchingReductionTBBLinearize<Transform> reduction(f);
    tbb::parallel_reduce(tbb::blocked_range<int>(0, num_points, 32), reduction);
    *H_target = reduction.H_target;
    *H_source = reduction.H_source;
    *H_target_source = reduction.H_target_source;
    *b_target = reduction.b_target;
    *b_source = reduction.b_source;
    return reduction.sum_errors;
  } else {
    ScanMatchingReductionTBBError<Transform> reduction(f);
    tbb::parallel_reduce(tbb::blocked_range<int>(0, num_points, 32), reduction);
    return reduction.sum_errors;
  }
}

#else
template <typename Transform>
double scan_matching_reduce_tbb(
  const Transform f,
  int num_points,
  Eigen::Matrix<double, 6, 6>* H_target,
  Eigen::Matrix<double, 6, 6>* H_source,
  Eigen::Matrix<double, 6, 6>* H_target_source,
  Eigen::Matrix<double, 6, 1>* b_target,
  Eigen::Matrix<double, 6, 1>* b_source) {
  std::cerr << "warning : TBB is not available" << std::endl;
  return scan_matching_reduce_omp(f, num_points, 1, H_target, H_source, H_target_source, b_target, b_source);
}
#endif

}  // namespace gtsam_points
