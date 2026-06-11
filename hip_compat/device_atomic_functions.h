// SPDX-License-Identifier: MIT
// Copyright (c) 2026  Advanced Micro Devices, Inc. (Jeff Daily <jeff.daily@amd.com>)
// HIP build shim: forwards the CUDA-toolkit header name to the gtsam_points
// CUDA-to-HIP compat header (HIP provides atomicCAS/atomicAdd/atomicMax via the
// HIP runtime). On the HIP include path (PRIVATE+BEFORE) only; on NVIDIA this
// directory is absent so the real toolkit header is used.
#pragma once
#include <gtsam_points/cuda/cuda_to_hip.h>
