// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/optimizers/fast_scatter.hpp>

#include <unordered_set>
#include <gtsam/inference/Ordering.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/linear/GaussianFactorGraph.h>

namespace gtsam_points {

FastScatter::FastScatter(const gtsam::GaussianFactorGraph& gfg, const gtsam::Ordering& ordering) {
  std::unordered_map<gtsam::Key, size_t> keymap;

  // If we have an ordering, pre-fill the ordered variables first
  for (gtsam::Key key : ordering) {
    keymap[key] = size();
    add(key, 0);
  }

  // Now, find dimensions of variables and/or extend
  for (const auto& factor : gfg) {
    if (!factor) continue;

    // TODO: Fix this hack to cope with zero-row Jacobians that come from BayesTreeOrphanWrappers
    const gtsam::JacobianFactor* asJacobian = dynamic_cast<const gtsam::JacobianFactor*>(factor.get());
    if (asJacobian && asJacobian->cols() <= 1) continue;

    // loop over variables
    for (auto variable = factor->begin(); variable != factor->end(); ++variable) {
      const gtsam::Key key = *variable;
      auto found = keymap.find(key);

      // iterator it = find(key);  // theoretically expensive, yet cache friendly
      if (found != keymap.end()) {
        auto it = begin() + found->second;
        it->dimension = factor->getDim(variable);
      } else {
        keymap[key] = size();
        add(key, factor->getDim(variable));
      }
    }
  }

  // To keep the same behavior as before, sort the keys after the ordering
  iterator first = begin();
  first += ordering.size();
  if (first != end()) std::sort(first, end());
}

}  // namespace gtsam_points
