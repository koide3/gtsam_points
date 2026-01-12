cmake_minimum_required(VERSION 3.22)
project(gtsam_points VERSION 1.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(BUILD_TESTS "Build tests" OFF)
option(BUILD_TESTS_PCL "Build tests that require PCL" OFF)
option(BUILD_DEMO "Build demo programs" OFF)
option(BUILD_EXAMPLE "Build example programs" OFF)
option(BUILD_TOOLS "Build tools" OFF)
option(BUILD_WITH_TBB "Build with TBB support" OFF)
option(BUILD_WITH_OPENMP "Build with OpenMP support" ON)
option(BUILD_WITH_CUDA "Build with GPU support" OFF)
option(BUILD_WITH_CUDA_MULTIARCH "Build with CUDA multi-architecture support" OFF)
option(BUILD_WITH_MARCH_NATIVE "Build with -march=native" OFF)
option(ENABLE_CPPCHECK "Enable cppcheck" OFF)
option(ENABLE_COVERAGE "Enable coverage check" OFF)

if(BUILD_WITH_MARCH_NATIVE)
  set(GTSAM_POINTS_WITH_MARCH_NATIVE 1)
  add_compile_options(-march=native)
  set(CMAKE_C_FLAGS "-march=native ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-march=native ${CMAKE_CXX_FLAGS}")
endif()

find_package(Boost REQUIRED COMPONENTS graph filesystem)
find_package(GTSAM 4.2 REQUIRED)
find_package(GTSAM_UNSTABLE 4.2 REQUIRED)
find_package(Eigen3 REQUIRED)

# Set optimizer source paths based on GTSAM version
if(GTSAM_VERSION VERSION_LESS "4.3")
  set(GTSAM_POINTS_ISAM2_SRC_DIR "gtsam4.2/")
  message(STATUS "Using ISAM2 sources for GTSAM4.2")
else()
  set(GTSAM_POINTS_ISAM2_SRC_DIR "")
  message(STATUS "Using default ISAM2 sources")
endif()

if(BUILD_WITH_TBB)
  find_package(TBB REQUIRED)
  set(GTSAM_POINTS_USE_TBB 1)
endif()

if(BUILD_WITH_OPENMP)
  find_package(OpenMP REQUIRED)
  set(GTSAM_POINTS_USE_OPENMP 1)
endif()

if(${BUILD_WITH_CUDA} AND ${EIGEN3_VERSION_STRING} AND ${EIGEN3_VERSION_STRING} VERSION_LESS "3.3.90")
  message(WARNING "Detected Eigen ${EIGEN3_VERSION_STRING} is not compatible with CUDA")
  message(WARNING "Use Eigen 3.3.90 or later (3.4.0 is recommended)")
endif()

# GPU-related
if(BUILD_WITH_CUDA)
  set(GTSAM_POINTS_USE_CUDA 1)
  add_definitions(-DBUILD_GTSAM_POINTS_GPU)

  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0")
    if(BUILD_WITH_CUDA_MULTIARCH)
      set(CMAKE_CUDA_ARCHITECTURES "all-major" CACHE STRING "CUDA architectures" FORCE)
    elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
      set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures" FORCE)
    endif()
  else()
    if(BUILD_WITH_CUDA_MULTIARCH)
      set(CMAKE_CUDA_ARCHITECTURES 75 80 87 89 90 CACHE STRING "CUDA architectures" FORCE)
    else()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
        message(WARNING "CMAKE_CUDA_ARCHITECTURES is not defined. Set to defaults")
        set(CMAKE_CUDA_ARCHITECTURES 75 80 87 89 90 CACHE STRING "CUDA architectures" FORCE)
      endif()
    endif()
  endif()
  message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
  set(CUDA_STANDARD 17)

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-c99-extensions")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

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

  set(GTSAM_POINTS_CUDA_VERSION ${CUDAToolkit_VERSION})
  set(GTSAM_POINTS_CUDA_VERSION_MAJOR ${CUDAToolkit_VERSION_MAJOR})
  set(GTSAM_POINTS_CUDA_VERSION_MINOR ${CUDAToolkit_VERSION_MINOR})
  set(GTSAM_POINTS_CUDA_VERSION_PATCH ${CUDAToolkit_VERSION_PATCH})
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

execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        OUTPUT_VARIABLE GTSAM_POINTS_GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
message(STATUS "GTSAM_POINTS_GIT_HASH=${GTSAM_POINTS_GIT_HASH}")

configure_file(include/gtsam_points/config.hpp.in include/gtsam_points/config.hpp)

###########
## Build ##
###########


add_library(gtsam_points SHARED
  # util
  src/gtsam_points/util/parallelism.cpp
  src/gtsam_points/util/bspline.cpp
  src/gtsam_points/util/continuous_trajectory.cpp
  src/gtsam_points/util/voxel_raycaster.cpp
  # features
  src/gtsam_points/features/covariance_estimation.cpp
  src/gtsam_points/features/normal_estimation.cpp
  src/gtsam_points/features/fpfh_estimation.cpp
  # registration
  src/gtsam_points/registration/alignment.cpp
  src/gtsam_points/registration/ransac.cpp
  src/gtsam_points/registration/graduated_non_convexity.cpp
  # segmentation
  src/gtsam_points/segmentation/min_cut.cpp
  src/gtsam_points/segmentation/region_growing.cpp
  # ann
  src/gtsam_points/ann/kdtree.cpp
  src/gtsam_points/ann/kdtreex.cpp
  src/gtsam_points/ann/intensity_kdtree.cpp
  src/gtsam_points/ann/ivox.cpp
  src/gtsam_points/ann/incremental_covariance_container.cpp
  src/gtsam_points/ann/incremental_covariance_voxelmap.cpp
  src/gtsam_points/ann/fast_occupancy_grid.cpp
  # types
  src/gtsam_points/types/offloadable.cpp
  src/gtsam_points/types/point_cloud.cpp
  src/gtsam_points/types/point_cloud_cpu.cpp
  src/gtsam_points/types/point_cloud_cpu_funcs.cpp
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
  src/gtsam_points/factors/reintegrated_imu_factor.cpp
  # experimental
  src/gtsam_points/factors/experimental/continuous_time_icp_factor.cpp
  # optimizers
  src/gtsam_points/optimizers/fast_scatter.cpp
  src/gtsam_points/optimizers/linearization_hook.cpp
  src/gtsam_points/optimizers/linear_system_builder.cpp
  src/gtsam_points/optimizers/gaussian_factor_graph_solver.cpp
  src/gtsam_points/optimizers/levenberg_marquardt_ext.cpp
  src/gtsam_points/optimizers/levenberg_marquardt_optimization_status.cpp
  src/gtsam_points/optimizers/${GTSAM_POINTS_ISAM2_SRC_DIR}isam2_ext.cpp
  src/gtsam_points/optimizers/${GTSAM_POINTS_ISAM2_SRC_DIR}isam2_ext_impl.cpp
  src/gtsam_points/optimizers/${GTSAM_POINTS_ISAM2_SRC_DIR}incremental_fixed_lag_smoother_ext.cpp
  src/gtsam_points/optimizers/incremental_fixed_lag_smoother_ext_with_fallback.cpp
  src/gtsam_points/optimizers/dogleg_optimizer_ext_impl.cpp
)
target_include_directories(gtsam_points PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/nanoflann/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(gtsam_points
  Boost::boost
  Boost::graph
  Boost::filesystem
  Eigen3::Eigen
  gtsam
  gtsam_unstable
  $<TARGET_NAME_IF_EXISTS:TBB::tbb>
  $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
)
set_target_properties(gtsam_points PROPERTIES
  VERSION   ${PROJECT_VERSION}
  SOVERSION 1
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
    # src/gtsam_points/cuda/gl_buffer_map.cu
    src/gtsam_points/cuda/nonlinear_factor_set_gpu.cpp
    src/gtsam_points/cuda/nonlinear_factor_set_gpu_create.cpp
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
    src/gtsam_points/factors/integrated_vgicp_factor_gpu.cpp
    # util
    src/gtsam_points/util/easy_profiler_cuda.cu
  )
  target_include_directories(gtsam_points_cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(gtsam_points_cuda
    CUDA::cudart
    Boost::boost
    Eigen3::Eigen
    gtsam
    gtsam_unstable
    $<TARGET_NAME_IF_EXISTS:TBB::tbb>
    $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
  )
  set_target_properties(gtsam_points_cuda PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION 1
  )

  target_link_libraries(gtsam_points
    gtsam_points_cuda
  )
endif()

#Demo
if(BUILD_DEMO)
  find_package(iridescence REQUIRED)

  file(GLOB demo_sources "src/demo/*.cpp")
  foreach(demo_src IN LISTS demo_sources)
    get_filename_component(demo_name ${demo_src} NAME_WE)

    add_executable(${demo_name} ${demo_src})
    target_link_libraries(${demo_name} gtsam_points Iridescence::Iridescence)
  endforeach()
endif()

#Example
if(BUILD_EXAMPLE)
  find_package(Iridescence REQUIRED)

  file(GLOB example_sources "src/example/*.cpp")
  foreach(example_src IN LISTS example_sources)
    get_filename_component(example_name ${example_src} NAME_WE)

    add_executable(${example_name} ${example_src})
    target_link_libraries(${example_name} gtsam_points Iridescence::Iridescence)
  endforeach()
endif()

#Example
if(BUILD_TOOLS)
  find_package(Iridescence REQUIRED)
  find_package(Boost REQUIRED COMPONENTS filesystem program_options)

  file(GLOB tool_sources "src/tools/*.cpp")
  foreach(tool_src IN LISTS tool_sources)
    get_filename_component(tool_name ${tool_src} NAME_WE)

    add_executable(${tool_name} ${tool_src})
    target_link_libraries(${tool_name} gtsam_points Iridescence::Iridescence Boost::program_options)
  endforeach()
endif()

# Test
if(BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
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

  if(BUILD_TESTS_PCL)
    enable_language(C)
    find_package(PCL REQUIRED)

    file(GLOB test_sources "src/test/pcl/*.cpp")
    foreach(test_src IN LISTS test_sources)
      get_filename_component(test_name ${test_src} NAME_WE)

      add_executable(${test_name} ${test_src})
      target_include_directories(${test_name} PRIVATE src/test/include ${Boost_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
      target_link_libraries(${test_name} gtsam_points gtest_main ${PCL_LIBRARIES})
      gtest_discover_tests(${test_name} WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    endforeach()
  endif()
endif()


#############
## Install ##
#############

include(GNUInstallDirs)
install(
  DIRECTORY include/ thirdparty/nanoflann/include/ ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

list(APPEND GTSAM_POINTS_LIBRARIES gtsam_points)
if(BUILD_WITH_CUDA)
  list(APPEND GTSAM_POINTS_LIBRARIES gtsam_points_cuda)
endif()

install(TARGETS ${GTSAM_POINTS_LIBRARIES}
  EXPORT gtsam_points-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
set(CMAKE_CONFIG_INSTALL_DIR
  "${CMAKE_INSTALL_LIBDIR}/cmake/gtsam_points"
  CACHE PATH "Install directory for CMake config files"
)
include(CMakePackageConfigHelpers)
install(EXPORT gtsam_points-targets
  FILE gtsam_points-targets.cmake
  NAMESPACE gtsam_points::
  DESTINATION ${CMAKE_CONFIG_INSTALL_DIR}
)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gtsam_points-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/gtsam_points-config.cmake"
  INSTALL_DESTINATION ${CMAKE_CONFIG_INSTALL_DIR}
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/gtsam_points-config-version.cmake"
  VERSION ${VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/gtsam_points-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/gtsam_points-config-version.cmake"
  DESTINATION ${CMAKE_CONFIG_INSTALL_DIR}
)

#############
## Package ##
#############

set(CPACK_GENERATOR "DEB")
set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
set(CPACK_SET_DESTDIR TRUE)
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "k.koide <k.koide@aist.go.jp>")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

set(GNUInstallDirs)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A collection of GTSAM factors for point cloud registration")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VENDOR "k.koide")

include(CPack)
