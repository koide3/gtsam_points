// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <cstdlib>

struct cudaGraphicsResource;

namespace gtsam_points {

class GLBufferMap {
public:
  enum BufferMode { NONE, READ_ONLY, WRITE_ONLY };

  GLBufferMap(size_t gl_id, BufferMode mode = NONE);
  ~GLBufferMap();

  size_t size() const { return buffer_size; }
  void* data() { return d_buffer; }

private:
  cudaGraphicsResource* gl_resource;

  size_t buffer_size;
  void* d_buffer;
};

}  // namespace gtsam_points
