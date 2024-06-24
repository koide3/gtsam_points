// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

struct CUgraph_st;
struct CUgraphExec_st;
struct CUstream_st;

namespace gtsam_points {

class CUDAGraphExec {
public:
  CUDAGraphExec(CUgraph_st* graph);
  ~CUDAGraphExec();

  CUDAGraphExec(const CUDAGraphExec&) = delete;
  CUDAGraphExec& operator=(const CUDAGraphExec&) = delete;

  void launch(CUstream_st* stream);

private:
  CUgraphExec_st* instance;
};

}  // namespace gtsam_points
