// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <Eigen/Sparse>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam_points/optimizers/fast_scatter.hpp>

namespace gtsam_points {

/**
 * @brief A helper class to build a dense linear system from a Gaussian factor graph.
 */
class DenseLinearSystemBuilder {
public:
  /**
   * @brief Constructor
   * @param gfg       Gaussian factor graph representing a linear sytem
   * @param ordering  Ordering
   */
  DenseLinearSystemBuilder(const gtsam::GaussianFactorGraph& gfg, const gtsam::Ordering& ordering);
  ~DenseLinearSystemBuilder();

  /**
   * @brief Convert a delta vector into a VectorValues
   */
  gtsam::VectorValues delta(const Eigen::VectorXd& x);

public:
  const gtsam_points::FastScatter scatter;

  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  double c;
};

/**
 * @brief A helper class to build a sparse linear system from a Gaussian factor graph.
 */
class SparseLinearSystemBuilderBase {
public:
  /**
   * @brief Constructor
   * @param gfg       Gaussian factor graph representing a linear sytem
   * @param ordering  Ordering
   */
  SparseLinearSystemBuilderBase(const gtsam::GaussianFactorGraph& gfg, const gtsam::Ordering& ordering);
  virtual ~SparseLinearSystemBuilderBase();

  /**
   * @brief Convert a delta vector into a VectorValues
   */
  gtsam::VectorValues delta(const Eigen::VectorXd& x);

public:
  const gtsam_points::FastScatter scatter;

  Eigen::SparseMatrix<double, Eigen::RowMajor> A;  // Lower triangular
  Eigen::VectorXd b;
  double c;
};

/**
 * @brief Sparse linear system builder with a block size option.
 * @note  If BLOCK_SIZE != -1, all variables in the Gaussian factor graph must have the dimension equals to BLOCK_SIZE.
 */
template <int BLOCK_SIZE = -1>
class SparseLinearSystemBuilder : public SparseLinearSystemBuilderBase {
public:
  SparseLinearSystemBuilder(const gtsam::GaussianFactorGraph& gfg, const gtsam::Ordering& ordering);
  virtual ~SparseLinearSystemBuilder() override {}
};

}  // namespace gtsam_points
