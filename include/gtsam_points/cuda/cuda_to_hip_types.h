// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)
// Copyright (c) 2026  Advanced Micro Devices, Inc. (Jeff Daily <jeff.daily@amd.com>)
//
// ROCm/HIP opaque-type bridge for the gtsam_points public API.
//
// The public headers use the CUDA driver opaque struct pointers (CUstream_st*,
// CUgraph_st*, ...) as their stream/graph/event types. On HIP those types are
// the ihip*/hipGraph* opaque structs. This header (force-included into BOTH the
// main gtsam_points library and the gtsam_points_cuda GPU library under USE_HIP)
// aliases the CUDA names to the matching HIP opaque structs so a CUstream_st*
// signature is identical to hipStream_t in every translation unit, keeping the
// host callers and the .cu definitions name-mangling compatible.
//
// It only forward-declares the HIP opaque structs (a pointer type needs no
// complete type), so it adds no HIP runtime dependency to host C++ files. The
// full cuda* runtime symbol aliases live in cuda_to_hip.h (GPU library only).
//
// Authored with the assistance of Claude (Anthropic).

#pragma once

#if defined(USE_HIP)

struct ihipStream_t;
struct ihipEvent_t;
struct ihipGraph;
struct hipGraphNode;
struct hipGraphExec;

#define CUstream_st ihipStream_t
#define CUevent_st ihipEvent_t
#define CUgraph_st ihipGraph
#define CUgraphNode_st hipGraphNode
#define CUgraphExec_st hipGraphExec

#endif
