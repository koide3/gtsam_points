cmake_minimum_required(VERSION 3.29)
project(gtsam_points LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(BUILD_TESTS "Build test" OFF)
option(BUILD_DEMO "Build demo programs" OFF)
option(BUILD_EXAMPLE "Build example programs" OFF)
option(BUILD_WITH_CUDA "Build with GPU support" OFF)
option(BUILD_WITH_CUDA_MULTIARCH "Build with CUDA multi-architecture support" OFF)
option(BUILD_WITH_MARCH_NATIVE "Build with -march=native" OFF)
option(ENABLE_CPPCHECK "Enable cppcheck" OFF)
option(ENABLE_COVERAGE "Enable coverage check" OFF)

if(BUILD_WITH_MARCH_NATIVE)
  add_compile_options(-march=native)
  set(CMAKE_C_FLAGS "-march=native ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-march=native ${CMAKE_CXX_FLAGS}")
endif()

find_package(Boost REQUIRED COMPONENTS timer filesystem)
find_package(GTSAM REQUIRED)

find_package(Eigen3 REQUIRED)
if(${BUILD_WITH_CUDA} AND ${EIGEN3_VERSION_STRING} VERSION_LESS "3.3.90")
  set(EIGEN3_INCLUDE_DIR "thirdparty/Eigen")
  message(STATUS "Use bundled Eigen because the system Eigen ${EIGEN3_VERSION_STRING} may not compatible with CUDA")
  message(STATUS "Override EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# GPU-related
if(BUILD_WITH_CUDA)
  add_definitions(-DBUILD_GTSAM_POINTS_GPU)

  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
  set(CUDA_STANDARD 17)

  if(BUILD_WITH_CUDA_MULTIARCH)
    set(CUDA_ARCHITECTURES "all-major")
  else()
    set(CUDA_ARCHITECTURES "navite")
  endif()

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-c99-extensions")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++14 --expt-relaxed-constexpr")

  if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
  else()
    # Suppress Eigen-related CUDA warnings
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=177")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=2739")

    if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL 11.2)
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20011")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20014")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=set_but_not_used")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
    endif()

    if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL 11.6)
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20236")
    endif()
  endif()
endif()

if(ENABLE_CPPCHECK)
  set(CMAKE_CXX_CPPCHECK "cppcheck")
  list(APPEND CMAKE_CXX_CPPCHECK
    "--enable=warning"
    "--inconclusive"
    "--force"
    "--inline-suppr"
    "--suppressions-list=${CMAKE_SOURCE_DIR}/.cppcheck_suppressions.txt"
  )
endif()

if(ENABLE_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -coverage")

  find_program(LCOV lcov REQUIRED)
  find_program(GENHTML genhtml REQUIRED)

  add_custom_target(coverage
    COMMAND ${LCOV} --directory . --capture --output-file coverage.info
    COMMAND ${GENHTML} --demangle-cpp -o coverage coverage.info
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

###########
## Build ##
###########

add_library(gtsam_points SHARED
  # util
  src/gtsam_points/util/covariance_estimation.cpp
  src/gtsam_points/util/normal_estimation.cpp
  src/gtsam_points/util/edge_plane_extraction.cpp
  src/gtsam_points/util/bspline.cpp
  src/gtsam_points/util/continuous_trajectory.cpp
  # ann
  src/gtsam_points/ann/kdtree.cpp
  src/gtsam_points/ann/intensity_kdtree.cpp
  src/gtsam_points/ann/ivox.cpp
  src/gtsam_points/ann/incremental_covariance_voxelmap.cpp
  # types
  src/gtsam_points/types/point_cloud.cpp
  src/gtsam_points/types/point_cloud_cpu.cpp
  src/gtsam_points/types/gaussian_voxelmap_cpu.cpp
  src/gtsam_points/types/gaussian_voxelmap_cpu_funcs.cpp
  # factors
  src/gtsam_points/factors/integrated_matching_cost_factor.cpp
  src/gtsam_points/factors/integrated_icp_factor.cpp
  src/gtsam_points/factors/integrated_gicp_factor.cpp
  src/gtsam_points/factors/integrated_vgicp_factor.cpp
  src/gtsam_points/factors/integrated_loam_factor.cpp
  src/gtsam_points/factors/intensity_gradients.cpp
  src/gtsam_points/factors/integrated_colored_gicp_factor.cpp
  src/gtsam_points/factors/integrated_color_consistency_factor.cpp
  src/gtsam_points/factors/integrated_ct_icp_factor.cpp
  src/gtsam_points/factors/integrated_ct_gicp_factor.cpp
  src/gtsam_points/factors/bundle_adjustment_factor_evm.cpp
  src/gtsam_points/factors/bundle_adjustment_factor_lsq.cpp
  # experimental
  src/gtsam_points/factors/experimental/continuous_time_icp_factor.cpp
  # optimizers
  src/gtsam_points/optimizers/fast_scatter.cpp
  src/gtsam_points/optimizers/linearization_hook.cpp
  src/gtsam_points/optimizers/linear_system_builder.cpp
  src/gtsam_points/optimizers/gaussian_factor_graph_solver.cpp
  src/gtsam_points/optimizers/levenberg_marquardt_ext.cpp
  src/gtsam_points/optimizers/levenberg_marquardt_optimization_status.cpp
  src/gtsam_points/optimizers/isam2_ext.cpp
  src/gtsam_points/optimizers/isam2_ext_impl.cpp
  src/gtsam_points/optimizers/incremental_fixed_lag_smoother_ext.cpp
  src/gtsam_points/optimizers/incremental_fixed_lag_smoother_ext_with_fallback.cpp
  # src/gtsam_points/optimizers/dogleg_optimizer_ext.cpp
  src/gtsam_points/optimizers/dogleg_optimizer_ext_impl.cpp
)
target_include_directories(gtsam_points PUBLIC
  include
  ${EIGEN3_INCLUDE_DIR}
  thirdparty/nanoflann/include
  ${Boost_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIRS}
)
target_link_libraries(gtsam_points
  pthread
  ${Boost_LIBRARIES}
  ${GTSAM_LIBRARIES}
)

# GPU-related
if(BUILD_WITH_CUDA)
  add_library(gtsam_points_cuda SHARED
    # cuda-related
    src/gtsam_points/cuda/check_error.cu
    src/gtsam_points/cuda/check_error_cusolver.cu
    src/gtsam_points/cuda/check_error_curand.cu
    src/gtsam_points/cuda/cuda_memory.cu
    src/gtsam_points/cuda/cuda_stream.cu
    src/gtsam_points/cuda/cuda_buffer.cu
    src/gtsam_points/cuda/cuda_device_sync.cu
    src/gtsam_points/cuda/cuda_device_prop.cu
    src/gtsam_points/cuda/cuda_graph.cu
    src/gtsam_points/cuda/cuda_graph_exec.cu
    src/gtsam_points/cuda/gl_buffer_map.cu
    src/gtsam_points/cuda/nonlinear_factor_set_gpu.cu
    src/gtsam_points/cuda/nonlinear_factor_set_gpu_create.cu
    src/gtsam_points/cuda/stream_roundrobin.cu
    src/gtsam_points/cuda/stream_temp_buffer_roundrobin.cu
    # types
    src/gtsam_points/types/point_cloud.cu
    src/gtsam_points/types/point_cloud_gpu.cu
    src/gtsam_points/types/gaussian_voxelmap_gpu.cu
    src/gtsam_points/types/gaussian_voxelmap_gpu_funcs.cu
    # factors
    src/gtsam_points/factors/integrated_vgicp_derivatives.cu
    src/gtsam_points/factors/integrated_vgicp_derivatives_inliers.cu
    src/gtsam_points/factors/integrated_vgicp_derivatives_compute.cu
    src/gtsam_points/factors/integrated_vgicp_derivatives_linearize.cu
    src/gtsam_points/factors/integrated_vgicp_factor_gpu.cu
    # util
    src/gtsam_points/util/easy_profiler_cuda.cu
  )
  target_include_directories(gtsam_points_cuda PUBLIC
    include
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
  )
  target_link_libraries(gtsam_points_cuda
    CUDA::cudart
  )
  set_target_properties(gtsam_points_cuda PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
  )

  target_link_libraries(gtsam_points
    gtsam_points_cuda
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
    target_link_libraries(${demo_name} gtsam_points ${Iridescence_LIBRARIES})
  endforeach()
endif()

#Example
if(BUILD_EXAMPLE)
  find_package(Iridescence REQUIRED)

  file(GLOB example_sources "src/example/*.cpp")
  foreach(example_src IN LISTS example_sources)
    get_filename_component(example_name ${example_src} NAME_WE)

    add_executable(${example_name} ${example_src})
    target_include_directories(${example_name} PUBLIC ${Iridescence_INCLUDE_DIRS})
    target_link_libraries(${example_name} gtsam_points ${Iridescence_LIBRARIES})
  endforeach()
endif()

# Test
if(BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  include(GoogleTest)

  file(GLOB test_sources "src/test/*.cpp")
  foreach(test_src IN LISTS test_sources)
    get_filename_component(test_name ${test_src} NAME_WE)

    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} gtsam_points gtest_main)
    target_include_directories(${test_name} PRIVATE ${Boost_INCLUDE_DIRS} src/test/include)
    gtest_discover_tests(${test_name} WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  endforeach()
endif()
