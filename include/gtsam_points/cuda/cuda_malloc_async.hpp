// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#if (CUDA_VERSION < 11000)

#warning "Use cuda(Malloc|Free) instead of cuda(Malloc|Free)Async for backward compatibility"

#define cudaMallocAsync(ptr, size, stream) cudaMalloc(ptr, size)
#define cudaFreeAsync(ptr, stream) cudaFree(ptr)

#endif