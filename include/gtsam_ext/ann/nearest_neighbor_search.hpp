#pragma once

#include <vector>

namespace gtsam_ext {

struct NearestNeighborSearch {
public:
  NearestNeighborSearch() {}
  virtual ~NearestNeighborSearch() {}

  virtual size_t knn_search(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists) const { return 0; };
};
}  // namespace gtsam_ext