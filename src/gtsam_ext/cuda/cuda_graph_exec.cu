// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_ext/cuda/cuda_graph_exec.hpp>

#include <gtsam_ext/cuda/check_error.cuh>

namespace gtsam_ext {

CUDAGraphExec::CUDAGraphExec(CUgraph_st* graph) {
  check_error << cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
}

CUDAGraphExec::~CUDAGraphExec() {
  check_error << cudaGraphExecDestroy(instance);
}

void CUDAGraphExec::launch(CUstream_st* stream) {
  check_error << cudaGraphLaunch(instance, stream);
}

}  // namespace gtsam_ext