cmake_minimum_required(VERSION 3.0.2)
project(gtsam_ext)

option(BUILD_TESTS "Build test" OFF)
option(BUILD_WITH_CUDA "Build with GPU support" OFF)
option(BUILD_WITH_MARCH_NATIVE "Build with -march=native" OFF)
option(BUILD_WITH_SYSTEM_EIGEN "Build with system-default Eigen" OFF)

if(BUILD_WITH_MARCH_NATIVE)
  add_definitions(-march=native)
  set(CMAKE_C_FLAGS "-march=native ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-march=native ${CMAKE_CXX_FLAGS}")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")
add_compile_options(-std=c++17)

find_package(Boost REQUIRED timer)
find_package(GTSAM REQUIRED)

if(BUILD_WITH_SYSTEM_EIGEN)
  find_package(Eigen3 REQUIRED)
else()
  set(EIGEN3_INCLUDE_DIR "thirdparty/Eigen")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# GPU-related
if(BUILD_WITH_CUDA)
  find_package(CUDA REQUIRED)
  add_definitions(-DBUILD_GTSAM_EXT_GPU)

  # set(CUDA_HOST_COMPILER clang++)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-c99-extensions")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++17 --expt-relaxed-constexpr")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_75")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_75,code=sm_75")
endif()

find_package(catkin REQUIRED COMPONENTS iridescence)

###########
## Build ##
###########

add_library(gtsam_ext STATIC
  # util
  src/gtsam_ext/util/covariance_estimation.cpp
  # types
  src/gtsam_ext/types/frame.cpp
  src/gtsam_ext/types/frame_cpu.cpp
  src/gtsam_ext/types/voxelized_frame.cpp
  src/gtsam_ext/types/voxelized_frame_cpu.cpp
  src/gtsam_ext/types/gaussian_voxelmap_cpu.cpp
  # factors
  src/gtsam_ext/factors/integrated_matching_cost_factor.cpp
  src/gtsam_ext/factors/integrated_icp_factor.cpp
  src/gtsam_ext/factors/integrated_gicp_factor.cpp
  src/gtsam_ext/factors/integrated_vgicp_factor.cpp
  src/gtsam_ext/factors/integrated_loam_factor.cpp
  src/gtsam_ext/factors/expression_icp_factor.cpp
  # optimizers
  src/gtsam_ext/optimizers/levenberg_marquardt_ext.cpp
  src/gtsam_ext/optimizers/levenberg_marquardt_optimization_status.cpp
  src/gtsam_ext/optimizers/isam2_ext.cpp
  src/gtsam_ext/optimizers/isam2_ext_impl.cpp
  src/gtsam_ext/optimizers/incremental_fixed_lag_smoother_ext.cpp
)
target_include_directories(gtsam_ext PRIVATE
  include
  ${EIGEN3_INCLUDE_DIR}
  thirdparty/nanoflann/include
  ${Boost_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIRS}
)
target_link_libraries(gtsam_ext
  pthread
  ${GTSAM_LIBRARIES}
)

# GPU-related
if(BUILD_WITH_CUDA)
  cuda_add_library(gtsam_ext_cuda STATIC
    # cuda-related
    src/gtsam_ext/cuda/nonlinear_factor_set_gpu.cu
    src/gtsam_ext/cuda/stream_roundrobin.cu
    src/gtsam_ext/cuda/stream_temp_buffer_roundrobin.cu
    # types
    src/gtsam_ext/types/frame.cu
    src/gtsam_ext/types/frame_gpu.cu
    src/gtsam_ext/types/voxelized_frame_gpu.cu
    src/gtsam_ext/types/gaussian_voxelmap_gpu.cu
    # factors
    src/gtsam_ext/factors/integrated_vgicp_derivatives.cu
    src/gtsam_ext/factors/integrated_vgicp_factor_gpu.cu
  )
  target_include_directories(gtsam_ext_cuda PUBLIC
    include
    ${EIGEN3_INCLUDE_DIR}
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

# test
if(BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)

  add_executable(test_matching_cost_factors src/test/test_matching_cost_factors.cpp)
  target_include_directories(test_matching_cost_factors PUBLIC
    include
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIRS}
  )
  target_link_libraries(test_matching_cost_factors
    gtsam_ext
    ${GTEST_LIBRARIES}
  )

  add_executable(test_loam_factors src/test/test_loam_factors.cpp)
  target_include_directories(test_loam_factors PUBLIC
    include
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIRS}
  )
  target_link_libraries(test_loam_factors
    gtsam_ext
    ${GTEST_LIBRARIES}
  )

  include(GoogleTest)
  gtest_discover_tests(test_matching_cost_factors
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    EXTRA_ARGS "${CMAKE_SOURCE_DIR}/data"
  )
  gtest_discover_tests(test_loam_factors
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    EXTRA_ARGS "${CMAKE_SOURCE_DIR}/data"
  )
endif()
