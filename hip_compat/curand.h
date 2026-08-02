// SPDX-License-Identifier: MIT
// Copyright (c) 2026  Advanced Micro Devices, Inc. (Jeff Daily <jeff.daily@amd.com>)
// HIP build shim: check_error_curand.cu includes <curand.h> only to name the
// CURAND_STATUS_* enum values for its diagnostic switch (the GPU library never
// calls curand). Forward to the compat header, which maps those names to the
// hipRAND HIPRAND_STATUS_* values. HIP include path (PRIVATE+BEFORE) only.
#pragma once
#include <gtsam_points/cuda/cuda_to_hip.h>
