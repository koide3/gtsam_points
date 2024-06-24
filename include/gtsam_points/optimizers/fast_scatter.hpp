/* ----------------------------------------------------------------------------   \
                                                                                             \ \
  * GTSAM Copyright 2010, Georgia Tech Research Corporation,                                   \
  * Atlanta, Georgia 30332-0415                                                                \
  * All Rights Reserved                                                                        \
  * Authors: Frank Dellaert, et al. (see THANKS for the full author list)                      \
                                                                                             \ \
  * See LICENSE for the license information                                                    \
                                                                                             \ \
  * -------------------------------------------------------------------------- */

/**
 * @file    Scatter.h
 * @brief   Maps global variable indices to slot indices
 * @author  Richard Roberts
 * @author  Frank Dellaert
 * @date    June 2015
 */

#pragma once

#include <gtsam/linear/Scatter.h>

namespace gtsam_points {

class FastScatter : public gtsam::Scatter {
public:
  FastScatter(const gtsam::GaussianFactorGraph& gfg, const gtsam::Ordering& ordering);
};

}  // namespace gtsam_points
