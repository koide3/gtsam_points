// SPDX-License-Identifier: MIT
// Copyright (c) 2026  Advanced Micro Devices, Inc. (Jeff Daily <jeff.daily@amd.com>)
// HIP build shim: maps the CUB include path to hipCUB. The compat header's
// `#define cub hipcub` rewrites the cub:: namespace in code; this maps the
// <cub/...> include path to its <hipcub/...> equivalent. HIP include path
// (PRIVATE+BEFORE) only; on NVIDIA this directory is absent so real CUB wins.
#pragma once
#include <hipcub/device/device_reduce.hpp>
