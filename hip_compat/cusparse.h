// SPDX-License-Identifier: MIT
// Copyright (c) 2026  Advanced Micro Devices, Inc. (Jeff Daily <jeff.daily@amd.com>)
// HIP build shim: check_error_cusolver.cu includes <cusparse.h> but references
// no cusparse symbol (its cuSOLVER status switch uses integer literals only),
// so this just satisfies the include. HIP include path (PRIVATE+BEFORE) only.
#pragma once
#include <gtsam_points/cuda/cuda_to_hip.h>
