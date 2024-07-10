// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <gtsam_points/cuda/check_error.cuh>
#include <gtsam_points/cuda/cuda_stream.hpp>
#include <gtsam_points/cuda/cuda_graph_exec.hpp>

struct CUgraph_st;
struct CUgraphNode_st;

namespace gtsam_points {
class CUDAGraph {
public:
  CUDAGraph();
  ~CUDAGraph();

  CUDAGraph(const CUDAGraph&) = delete;
  CUDAGraph& operator=(const CUDAGraph&) = delete;

  template <typename Func>
  CUgraphNode_st* add_kernel(const Func&& func) {
    gtsam_points::CUDAStream stream;
    CUgraph_st* sub_graph;
    check_error << cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    func(stream);

    check_error << cudaStreamEndCapture(stream, &sub_graph);

    CUgraphNode_st* node;
    check_error << cudaGraphAddChildGraphNode(&node, graph, nullptr, 0, sub_graph);

    return node;
  }

  void add_dependency(CUgraphNode_st* from, CUgraphNode_st* to);

  std::shared_ptr<CUDAGraphExec> instantiate();

private:
  CUgraph_st* graph;
};

}  // namespace gtsam_points
