// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <cuda.h>
#include <gtsam_points/cuda/cuda_graph.cuh>

namespace gtsam_points {

CUDAGraph::CUDAGraph() {
  check_error << cudaGraphCreate(&graph, 0);
}

CUDAGraph::~CUDAGraph() {
  check_error << cudaGraphDestroy(graph);
}

void CUDAGraph::add_dependency(CUgraphNode_st* from, CUgraphNode_st* to) {
#if CUDA_VERSION >= 13000
  check_error << cudaGraphAddDependencies(graph, &from, &to, nullptr, 1);
#else
  check_error << cudaGraphAddDependencies(graph, &from, &to, 1);
#endif
}

std::shared_ptr<CUDAGraphExec> CUDAGraph::instantiate() {
  return std::make_shared<CUDAGraphExec>(graph);
}

}  // namespace gtsam_points
