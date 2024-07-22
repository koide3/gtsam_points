// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/cuda/gl_buffer_map.hpp>

#include <GL/gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <gtsam_points/cuda/check_error.cuh>

namespace gtsam_points {

GLBufferMap::GLBufferMap(size_t gl_id, BufferMode mode) {
  unsigned int flags = 0;

  switch (mode) {
    case GLBufferMap::NONE:
      flags = cudaGraphicsRegisterFlagsNone;
      break;
    case GLBufferMap::READ_ONLY:
      flags = cudaGraphicsRegisterFlagsReadOnly;
      break;
    case GLBufferMap::WRITE_ONLY:
      flags = cudaGraphicsRegisterFlagsWriteDiscard;
      break;
  }

  check_error << cudaGraphicsGLRegisterBuffer(&gl_resource, gl_id, flags);
  check_error << cudaGraphicsMapResources(1, &gl_resource);
  check_error << cudaGraphicsResourceGetMappedPointer(&d_buffer, &buffer_size, gl_resource);
}

GLBufferMap::~GLBufferMap() {
  check_error << cudaGraphicsUnmapResources(1, &gl_resource);
  check_error << cudaGraphicsUnregisterResource(gl_resource);
}

}  // namespace gtsam_points
