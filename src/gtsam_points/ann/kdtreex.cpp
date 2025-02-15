// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/ann/kdtreex.hpp>
#include <gtsam_points/ann/impl/kdtreex_impl.hpp>

namespace gtsam_points {

template class KdTreeX<-1>;
template class KdTreeX<4>;
template class KdTreeX<33>;
template class KdTreeX<125>;

}  // namespace gtsam_points
