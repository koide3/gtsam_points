// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

namespace gtsam_ext {

/**
 * @brief Synchronize all CUDA devices
 */
void cuda_device_synchronize();

}