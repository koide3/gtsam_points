cmake_minimum_required(VERSION 3.0.2)
project(gtsam_ext)

option(BUILD_GTSAM_EXT_GPU "Build with GPU support" ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

add_compile_options(-std=c++17)

find_package(Boost REQUIRED timer)
find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# GPU-related
if(BUILD_GTSAM_EXT_GPU)
  find_package(CUDA REQUIRED)
  add_definitions(-DBUILD_GTSAM_EXT_GPU)

  # set(CUDA_HOST_COMPILER clang++)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-c99-extensions")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++17 --expt-relaxed-constexpr")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_75")
  # set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_60,code=sm_60")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_75,code=sm_75")
endif()

find_package(catkin REQUIRED COMPONENTS iridescence)

###################################
## catkin specific configuration ##
###################################
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES gtsam_ext
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

add_library(gtsam_ext SHARED
  # types
  src/gtsam_ext/types/frame.cpp
  src/gtsam_ext/types/voxelized_frame.cpp
  src/gtsam_ext/types/voxelized_frame_cpu.cpp
  src/gtsam_ext/types/gaussian_voxelmap_cpu.cpp
  # factors
  src/gtsam_ext/factors/integrated_matching_cost_factor.cpp
  src/gtsam_ext/factors/integrated_icp_factor.cpp
  src/gtsam_ext/factors/integrated_gicp_factor.cpp
  src/gtsam_ext/factors/integrated_vgicp_factor.cpp
  # optimizers
  src/gtsam_ext/optimizers/levenberg_marquardt_ext.cpp
  src/gtsam_ext/optimizers/levenberg_marquardt_optimization_status.cpp
)
target_include_directories(gtsam_ext PRIVATE
  include
  thirdparty/nanoflann/include
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIRS}
)
target_link_libraries(gtsam_ext
  ${GTSAM_LIBRARIES}
)

add_executable(test_cpu
  src/test/test_cpu.cpp
)
target_include_directories(test_cpu PRIVATE
  include
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)
target_link_libraries(test_cpu
  gtsam_ext
  gtsam_ext_cuda
  ${catkin_LIBRARIES}
)


# GPU-related
if(BUILD_GTSAM_EXT_GPU)
  cuda_add_library(gtsam_ext_cuda SHARED
    # cuda-related
    src/gtsam_ext/cuda/nonlinear_factor_set_gpu.cu
    src/gtsam_ext/cuda/stream_roundrobin.cu
    src/gtsam_ext/cuda/stream_temp_buffer_roundrobin.cu
    # types
    src/gtsam_ext/types/frame.cu
    src/gtsam_ext/types/voxelized_frame_gpu.cu
    src/gtsam_ext/types/gaussian_voxelmap_gpu.cu
    # factors
    src/gtsam_ext/factors/integrated_vgicp_derivatives.cu
  )
  target_include_directories(gtsam_ext_cuda PUBLIC
    include
    thirdparty/Eigen
    ${Boost_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
  )
  target_link_directories(gtsam_ext_cuda PUBLIC
    ${CUDA_LIBRARY_DIRS}
  )
  target_link_libraries(gtsam_ext
    gtsam_ext_cuda
  )
endif()
