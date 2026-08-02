// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)
// Copyright (c) 2026  Advanced Micro Devices, Inc. (Jeff Daily <jeff.daily@amd.com>)
//
// ROCm/HIP compatibility shim for the gtsam_points GPU library.
//
// This is the only file that knows about HIP. On AMD (USE_HIP) it pulls in the
// HIP runtime and aliases the CUDA spellings the GPU library actually uses to
// their HIP equivalents; on NVIDIA it is a plain include of the CUDA runtime.
// It is force-included into every gtsam_points_cuda translation unit by the
// build (so source files keep their CUDA spelling and the diff stays small).
//
// Authored with the assistance of Claude (Anthropic).

#pragma once

#if defined(USE_HIP)

// libc host string/memory decls must win over HIP's __device__ overloads of
// memcpy/memset, which become visible once hip_runtime.h is in scope inside a
// translation unit compiled as HIP (point_cloud_gpu.cu uses host std::memcpy).
#include <cstring>
#include <cstdlib>

#include <hip/hip_runtime.h>

// The GPU library uses cudaMallocAsync/cudaFreeAsync (stream-ordered) and the
// CUDA-12 graph-instantiate / 4-arg add-dependencies arities. CUDA_VERSION is
// undefined on HIP, which would (a) trip cuda_malloc_async.hpp's
// `#if (CUDA_VERSION < 11000)` and silently downgrade every async alloc to a
// synchronous hipMalloc, and (b) leave the cuda_graph.cu arity branch on the
// wrong side. Pin it into [11000, 13000): hipMallocAsync stays the chosen path
// and the 4-arg hipGraphAddDependencies / 5-arg hipGraphInstantiate signatures
// (ROCm 7.x) are selected.
#ifndef CUDA_VERSION
#define CUDA_VERSION 12000
#endif

// Opaque API-type bridge (CUstream_st/CUgraph_st/... -> HIP opaque structs).
// Kept in a separate header so it is also force-included into the main library,
// keeping a CUstream_st* signature name-mangling compatible across both libs.
#include <gtsam_points/cuda/cuda_to_hip_types.h>

// hipCUB provides the cub:: device primitives under the hipcub:: namespace.
#define cub hipcub

// Error handling / device query
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMemGetInfo hipMemGetInfo

// Memory (stream-ordered async allocation is used pervasively)
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocAsync hipMallocAsync
#define cudaFreeAsync hipFreeAsync
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaHostRegister hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaHostRegisterDefault hipHostRegisterDefault
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

// Streams
#define cudaStream_t hipStream_t
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal

// Events (easy_profiler timing)
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventDestroy hipEventDestroy

// CUDA Graphs (compile parity; the VGICP gate drives streams + hipCUB directly)
#define cudaGraphCreate hipGraphCreate
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphAddDependencies hipGraphAddDependencies
#define cudaGraphAddChildGraphNode hipGraphAddChildGraphNode
#define cudaGraphInstantiate hipGraphInstantiate
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphLaunch hipGraphLaunch

// rocThrust exposes its device backend execution policies under thrust::hip
// (thrust::cuda::par[_nosync] is the CUDA backend and is not valid here).
// Pull the HIP execution-policy header so thrust::hip is defined, then alias
// the namespace so the ~12 thrust::cuda::par[_nosync].on(stream) call sites
// resolve to the HIP backend unchanged. This header is force-included at the
// top of every TU, so the alias must not depend on the source's own includes.
// Guard on __HIPCC__: only the .cu sources are compiled -x hip (and use the
// thrust policy); the host .cpp in the GPU lib are compiled by the CXX compiler
// (which cannot parse rocThrust/rocPRIM) and never use thrust::cuda::par.
#if defined(__HIPCC__)
#include <thrust/system/hip/execution_policy.h>
namespace thrust {
namespace cuda = hip;
}
#endif

// curand error-name strings (check_error_curand.cu): hipRAND ships the same
// status names under the HIPRAND_STATUS_* spelling. The GPU library never calls
// curand at runtime (only this diagnostic switch), so no hipRAND link is needed.
#include <hiprand/hiprand.h>
#define CURAND_STATUS_SUCCESS HIPRAND_STATUS_SUCCESS
#define CURAND_STATUS_VERSION_MISMATCH HIPRAND_STATUS_VERSION_MISMATCH
#define CURAND_STATUS_NOT_INITIALIZED HIPRAND_STATUS_NOT_INITIALIZED
#define CURAND_STATUS_ALLOCATION_FAILED HIPRAND_STATUS_ALLOCATION_FAILED
#define CURAND_STATUS_TYPE_ERROR HIPRAND_STATUS_TYPE_ERROR
#define CURAND_STATUS_OUT_OF_RANGE HIPRAND_STATUS_OUT_OF_RANGE
#define CURAND_STATUS_LENGTH_NOT_MULTIPLE HIPRAND_STATUS_LENGTH_NOT_MULTIPLE
#define CURAND_STATUS_DOUBLE_PRECISION_REQUIRED HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED
#define CURAND_STATUS_LAUNCH_FAILURE HIPRAND_STATUS_LAUNCH_FAILURE
#define CURAND_STATUS_PREEXISTING_FAILURE HIPRAND_STATUS_PREEXISTING_FAILURE
#define CURAND_STATUS_INITIALIZATION_FAILED HIPRAND_STATUS_INITIALIZATION_FAILED
#define CURAND_STATUS_ARCH_MISMATCH HIPRAND_STATUS_ARCH_MISMATCH
#define CURAND_STATUS_INTERNAL_ERROR HIPRAND_STATUS_INTERNAL_ERROR

#else

#include <cuda_runtime.h>

#endif
