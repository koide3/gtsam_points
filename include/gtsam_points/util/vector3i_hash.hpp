// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <Eigen/Core>
#include <boost/functional/hash/hash.hpp>

namespace gtsam_points {

/**
 * @brief Spatial hashing function using boost::hash_combine
 */
class Vector3iHash {
public:
  size_t operator()(const Eigen::Vector3i& x) const {
    size_t seed = 0;
    boost::hash_combine(seed, x[0]);
    boost::hash_combine(seed, x[1]);
    boost::hash_combine(seed, x[2]);
    return seed;
  }
};

/**
 * @brief Spatial hashing function
 *        Teschner et al., "Optimized Spatial Hashing for Collision Detection of Deformable Objects", VMV2003
 */
class XORVector3iHash {
public:
  size_t operator()(const Eigen::Vector3i& x) const {
    const size_t p1 = 9132043225175502913;
    const size_t p2 = 7277549399757405689;
    const size_t p3 = 6673468629021231217;
    return static_cast<size_t>((x[0] * p1) ^ (x[1] * p2) ^ (x[2] * p3));
  }
};

}  // namespace gtsam_points