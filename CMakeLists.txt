cmake_minimum_required(VERSION 3.0.2)
project(gtsam_ext)

add_compile_options(-std=c++17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(BUILD_TESTS "Build test" OFF)
option(BUILD_DEMO "Build demo programs" OFF)
option(BUILD_WITH_CUDA "Build with GPU support" OFF)
option(BUILD_WITH_MARCH_NATIVE "Build with -march=native" OFF)
option(BUILD_WITH_SYSTEM_EIGEN "Build with system-default Eigen" OFF)

if(BUILD_WITH_MARCH_NATIVE)
  add_definitions(-march=native)
  set(CMAKE_C_FLAGS "-march=native ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-march=native ${CMAKE_CXX_FLAGS}")
endif()

find_package(Boost REQUIRED COMPONENTS timer filesystem)
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
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcudafe --diag_suppress=2739")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcudafe --diag_suppress=set_but_not_used")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
endif()

###########
## Build ##
###########

add_library(gtsam_ext SHARED
  # util
  src/gtsam_ext/util/covariance_estimation.cpp
  src/gtsam_ext/util/normal_estimation.cpp
  src/gtsam_ext/util/edge_plane_extraction.cpp
  src/gtsam_ext/util/bspline.cpp
  src/gtsam_ext/util/continuous_trajectory.cpp
  # ann
  src/gtsam_ext/ann/kdtree.cpp
  src/gtsam_ext/ann/intensity_kdtree.cpp
  src/gtsam_ext/ann/ivox.cpp
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
  src/gtsam_ext/factors/intensity_gradients.cpp
  src/gtsam_ext/factors/integrated_colored_gicp_factor.cpp
  src/gtsam_ext/factors/integrated_color_consistency_factor.cpp
  src/gtsam_ext/factors/integrated_ct_icp_factor.cpp
  src/gtsam_ext/factors/integrated_ct_gicp_factor.cpp
  src/gtsam_ext/factors/bundle_adjustment_factor_evm.cpp
  src/gtsam_ext/factors/bundle_adjustment_factor_lsq.cpp
  # experimental
  src/gtsam_ext/factors/experimental/continuous_time_icp_factor.cpp
  # optimizers
  src/gtsam_ext/optimizers/levenberg_marquardt_ext.cpp
  src/gtsam_ext/optimizers/levenberg_marquardt_optimization_status.cpp
  src/gtsam_ext/optimizers/isam2_ext.cpp
  src/gtsam_ext/optimizers/isam2_ext_impl.cpp
  src/gtsam_ext/optimizers/incremental_fixed_lag_smoother_ext.cpp
  # src/gtsam_ext/optimizers/dogleg_optimizer_ext.cpp
  src/gtsam_ext/optimizers/dogleg_optimizer_ext_impl.cpp
)
target_include_directories(gtsam_ext PUBLIC
  include
  ${EIGEN3_INCLUDE_DIR}
  thirdparty/nanoflann/include
  ${Boost_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIRS}
)
target_link_libraries(gtsam_ext
  pthread
  ${Boost_LIBRARIES}
  ${GTSAM_LIBRARIES}
)

# GPU-related
if(BUILD_WITH_CUDA)
  cuda_add_library(gtsam_ext_cuda SHARED
    # cuda-related
    src/gtsam_ext/cuda/cuda_device_sync.cu
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

#Demo
if(BUILD_DEMO)
  find_package(Iridescence REQUIRED)

  file(GLOB demo_sources "src/demo/*.cpp")
  foreach(demo_src IN LISTS demo_sources)
    get_filename_component(demo_name ${demo_src} NAME_WE)

    add_executable(${demo_name} ${demo_src})
    target_include_directories(${demo_name} PUBLIC ${Iridescence_INCLUDE_DIRS})
    target_link_libraries(${demo_name} gtsam_ext ${Iridescence_LIBRARIES})
  endforeach()
endif()

# Test
if(BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  include(GoogleTest)

  file(GLOB test_sources "src/test/*.cpp")
  foreach(test_src IN LISTS test_sources)
    get_filename_component(test_name ${test_src} NAME_WE)

    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} gtsam_ext ${GTEST_LIBRARIES})
    gtest_discover_tests(${test_name} WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  endforeach()
endif()
