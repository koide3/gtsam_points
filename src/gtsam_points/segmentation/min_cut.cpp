// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/segmentation/min_cut.hpp>
#include <gtsam_points/segmentation/impl/min_cut_impl.hpp>

namespace gtsam_points {

template MinCutResult
min_cut_(const PointCloud& points, const NearestNeighborSearch& search, size_t source_pt_index, size_t sink_pt_index, const MinCutParams& params);
}