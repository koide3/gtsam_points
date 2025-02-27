// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/optimizers/linear_system_builder.hpp>

#include <numeric>
#include <unordered_map>
#include <gtsam/linear/HessianFactor.h>

#include <gtsam_points/config.hpp>
#include <gtsam_points/util/parallelism.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/concurrent_hash_map.h>
#endif

namespace {

template <typename T>
struct PairHasher {
public:
  size_t operator()(const std::pair<T, T>& x) const {
    const size_t p1 = 73856093;
    const size_t p2 = 19349669;  // 19349663 was not a prime number
    return static_cast<size_t>((x.first * p1) ^ (x.second * p2));
  }
};

struct BlockSlot {
  size_t dim;
  size_t index;
};

}  // namespace

namespace gtsam_points {

DenseLinearSystemBuilder::DenseLinearSystemBuilder(const gtsam::GaussianFactorGraph& gfg, const gtsam::Ordering& ordering) : scatter(gfg, ordering) {
  gtsam::HessianFactor combined(gfg, scatter);

  gtsam::Matrix augmented = combined.info().selfadjointView();

  const int N = augmented.rows() - 1;
  A = augmented.topLeftCorner(N, N);
  b = augmented.topRightCorner(N, 1);
  c = augmented(N, N);
}

DenseLinearSystemBuilder::~DenseLinearSystemBuilder() {}

gtsam::VectorValues DenseLinearSystemBuilder::delta(const Eigen::VectorXd& x) {
  return gtsam::VectorValues(x, scatter);
}

SparseLinearSystemBuilderBase::SparseLinearSystemBuilderBase(const gtsam::GaussianFactorGraph& gfg, const gtsam::Ordering& ordering)
: scatter(gfg, ordering) {}

SparseLinearSystemBuilderBase::~SparseLinearSystemBuilderBase() {}

gtsam::VectorValues SparseLinearSystemBuilderBase::delta(const Eigen::VectorXd& x) {
  return gtsam::VectorValues(x, scatter);
}

template <int BLOCK_SIZE>
SparseLinearSystemBuilder<BLOCK_SIZE>::SparseLinearSystemBuilder(const gtsam::GaussianFactorGraph& gfg, const gtsam::Ordering& ordering)
: SparseLinearSystemBuilderBase(gfg, ordering) {
  //
  using BlockMatrix = Eigen::Matrix<double, BLOCK_SIZE, BLOCK_SIZE>;
  using BlockVector = Eigen::Matrix<double, BLOCK_SIZE, 1>;

  std::unordered_map<gtsam::Key, BlockSlot> block_slots;
  size_t total_dim = 0;
  for (const auto& slot : scatter) {
    block_slots[slot.key] = {slot.dimension, total_dim};
    total_dim += slot.dimension;

    if (BLOCK_SIZE > 0 && BLOCK_SIZE != slot.dimension) {
      std::cerr << "block size mismatch!!" << std::endl;
    }
  }

  c = 0.0;

  if (is_omp_default()) {
    std::unordered_map<std::pair<int, int>, BlockMatrix, PairHasher<int>> A_blocks;
    std::unordered_map<int, BlockVector> b_blocks;

    for (const auto& gf : gfg) {
      std::vector<BlockSlot> slots(gf->keys().size());
      std::transform(gf->keys().begin(), gf->keys().end(), slots.begin(), [&](const gtsam::Key key) { return block_slots[key]; });

      const auto augmented = gf->augmentedInformation();
      const int N = augmented.rows() - 1;

      c += augmented(augmented.rows() - 1, augmented.cols() - 1);

      for (int i = 0, index_i = 0; i < gf->keys().size(); index_i += slots[i++].dim) {
        const int dim_i = slots[i].dim;
        const int global_index_i = slots[i].index;

        BlockVector b_i;
        if constexpr (BLOCK_SIZE < 0) {
          b_i = augmented.block(index_i, N, dim_i, 1);
        } else {
          b_i = augmented.block<BLOCK_SIZE, 1>(index_i, N);
        }

        auto found = b_blocks.find(global_index_i);
        if (found == b_blocks.end()) {
          b_blocks.emplace_hint(found, global_index_i, b_i);
        } else {
          found->second += b_i;
        }
      }

      for (int i = 0, index_i = 0; i < gf->keys().size(); index_i += slots[i++].dim) {
        for (int j = i, index_j = index_i; j < gf->keys().size(); index_j += slots[j++].dim) {
          const int dim_i = slots[i].dim;
          const int dim_j = slots[j].dim;
          int global_index_i = slots[i].index;
          int global_index_j = slots[j].index;

          BlockMatrix H_ij;
          if constexpr (BLOCK_SIZE < 0) {
            H_ij = augmented.block(index_i, index_j, dim_i, dim_j);
          } else {
            H_ij = augmented.block<BLOCK_SIZE, BLOCK_SIZE>(index_i, index_j);
          }

          if (global_index_i < global_index_j) {
            std::swap(global_index_i, global_index_j);
            H_ij = H_ij.transpose().eval();
          }

          auto found = A_blocks.find({global_index_i, global_index_j});
          if (found == A_blocks.end()) {
            A_blocks.emplace_hint(found, std::make_pair(global_index_i, global_index_j), H_ij);
          } else {
            found->second += H_ij;
          }
        }
      }
    }

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(A_blocks.size() * 6 * 6);

    for (const auto& [index, H_ij] : A_blocks) {
      const auto [index_i, index_j] = index;
      Eigen::SparseMatrix<double> sH_ij = H_ij.sparseView();

      for (int k = 0; k < sH_ij.outerSize(); k++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(sH_ij, k); it; ++it) {
          triplets.emplace_back(index_i + it.row(), index_j + it.col(), it.value());
        }
      }
    }

    A.resize(total_dim, total_dim);
    A.setFromTriplets(triplets.begin(), triplets.end());

    b.resize(total_dim);
    for (const auto& [index_i, b_i] : b_blocks) {
      b.block(index_i, 0, b_i.size(), 1) = b_i;
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    using BlockMatrixMap = tbb::concurrent_hash_map<std::uint64_t, BlockMatrix>;
    using BlockVectorMap = tbb::concurrent_hash_map<std::uint64_t, BlockVector>;
    BlockMatrixMap A_blocks;
    BlockVectorMap b_blocks;
    std::vector<double> all_c(gfg.size(), 0.0);

    const auto perfactor_task = [&](const gtsam::GaussianFactor& gf) {
      std::vector<BlockSlot> slots(gf.keys().size());
      std::transform(gf.keys().begin(), gf.keys().end(), slots.begin(), [&](const gtsam::Key key) { return block_slots[key]; });

      const auto augmented = gf.augmentedInformation();
      const int N = augmented.rows() - 1;

      // Add b
      for (int i = 0, index_i = 0; i < gf.keys().size(); index_i += slots[i++].dim) {
        const int dim_i = slots[i].dim;
        const int global_index_i = slots[i].index;

        BlockVector b_i;
        if constexpr (BLOCK_SIZE < 0) {
          b_i = augmented.block(index_i, N, dim_i, 1);
        } else {
          b_i = augmented.block<BLOCK_SIZE, 1>(index_i, N);
        }

        typename BlockVectorMap::accessor a;
        if (!b_blocks.insert(a, global_index_i)) {
          a->second += b_i;
        } else {
          a->second = b_i;
        }
      }

      // Add A
      for (int i = 0, index_i = 0; i < gf.keys().size(); index_i += slots[i++].dim) {
        for (int j = i, index_j = index_i; j < gf.keys().size(); index_j += slots[j++].dim) {
          const int dim_i = slots[i].dim;
          const int dim_j = slots[j].dim;
          int global_index_i = slots[i].index;
          int global_index_j = slots[j].index;

          BlockMatrix H_ij;
          if constexpr (BLOCK_SIZE < 0) {
            H_ij = augmented.block(index_i, index_j, dim_i, dim_j);
          } else {
            H_ij = augmented.block<BLOCK_SIZE, BLOCK_SIZE>(index_i, index_j);
          }

          if (global_index_i < global_index_j) {
            std::swap(global_index_i, global_index_j);
            H_ij = H_ij.transpose().eval();
          }

          const size_t key = (static_cast<std::uint64_t>(global_index_i) << 32) | global_index_j;

          typename BlockMatrixMap::accessor a;
          if (!A_blocks.insert(a, key)) {
            a->second += H_ij;
          } else {
            a->second = H_ij;
          }
        }
      }

      return augmented(augmented.rows() - 1, augmented.cols() - 1);
    };

    tbb::parallel_for(static_cast<size_t>(0), gfg.size(), [&](size_t i) { all_c[i] = perfactor_task(*gfg[i]); });
    c = std::accumulate(all_c.begin(), all_c.end(), 0.0);

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(A_blocks.size() * 6 * 6);

    for (auto itr = A_blocks.begin(); itr != A_blocks.end(); itr++) {
      const size_t key = itr->first;
      const int index_i = key >> 32;
      const int index_j = key & ((1ull << 32) - 1);
      const BlockMatrix& H_ij = itr->second;

      Eigen::SparseMatrix<double> sH_ij = H_ij.sparseView();

      for (int k = 0; k < sH_ij.outerSize(); k++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(sH_ij, k); it; ++it) {
          triplets.emplace_back(index_i + it.row(), index_j + it.col(), it.value());
        }
      }
    }

    A.resize(total_dim, total_dim);
    A.setFromTriplets(triplets.begin(), triplets.end());

    b.resize(total_dim);
    for (auto itr = b_blocks.begin(); itr != b_blocks.end(); itr++) {
      b.block(itr->first, 0, itr->second.size(), 1) = itr->second;
    }

#else
    std::cerr << "error : gtsam_points is not built with TBB!!" << std::endl;
#endif
  }
}

template class SparseLinearSystemBuilder<-1>;
template class SparseLinearSystemBuilder<3>;
template class SparseLinearSystemBuilder<6>;
}  // namespace gtsam_points
