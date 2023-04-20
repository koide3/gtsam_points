// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam_ext/optimizers/linearization_hook.hpp>

namespace gtsam_ext {

std::shared_ptr<NonlinearFactorSet> create_nonlinear_factor_set_gpu();

}  // namespace gtsam_ext
