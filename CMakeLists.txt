cmake_minimum_required(VERSION 3.22)
project(gtsam_points VERSION 1.2.2 LANGUAGES CXX)

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
option(BUILD_WITH_HIP "Build with ROCm/HIP GPU support (AMD)" OFF)
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

# GPU-related (ROCm/HIP for AMD). Mirrors the BUILD_WITH_CUDA arm above: keeps
# GTSAM_POINTS_USE_CUDA / BUILD_GTSAM_POINTS_GPU defined so all existing GPU
# guards compile unchanged, and builds the same gtsam_points_cuda source list
# with the .cu sources marked LANGUAGE HIP (see the GPU library section).
if(BUILD_WITH_HIP)
  set(GTSAM_POINTS_USE_CUDA 1)
  set(GTSAM_POINTS_USE_HIP 1)
  add_definitions(-DBUILD_GTSAM_POINTS_GPU)

  # Windows/clang needs _USE_MATH_DEFINES for M_PI and NOMINMAX to prevent
  # the Windows min/max macros from colliding with std::min/max.
  if(WIN32)
    add_compile_definitions(_USE_MATH_DEFINES NOMINMAX)
  endif()

  find_package(hip REQUIRED)
  # enable_language(HIP) honors -DCMAKE_HIP_ARCHITECTURES, otherwise auto-detects
  # the host GPU and errors on a no-GPU build host.
  enable_language(HIP)
  message(STATUS "CMAKE_HIP_ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")

  set(GTSAM_POINTS_CUDA_VERSION 0)
  set(GTSAM_POINTS_CUDA_VERSION_MAJOR 0)
  set(GTSAM_POINTS_CUDA_VERSION_MINOR 0)
  set(GTSAM_POINTS_CUDA_VERSION_PATCH 0)
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


if(WIN32 AND BUILD_WITH_HIP)
  # On Windows with a static GTSAM (GTSAM_EXPORT = empty, no dllimport), both
  # gtsam_points and gtsam_points_cuda are built as STATIC to avoid circular DLL
  # import issues (PointCloudGPU inherits PointCloudCPU across library boundary).
  # Test executables link all three static libs directly.
  set(_gtsam_points_type STATIC)
else()
  set(_gtsam_points_type SHARED)
endif()
add_library(gtsam_points ${_gtsam_points_type}
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
if(NOT WIN32 OR NOT BUILD_WITH_HIP)
  set_target_properties(gtsam_points PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION 1
  )
endif()

if(BUILD_WITH_HIP)
  # The public API uses CUstream_st* etc. The main library's host .cpp call into
  # the GPU library across that type, so they must see the same HIP opaque-type
  # bridge (CUstream_st -> ihipStream_t) for name mangling to match. Force-
  # include the lightweight type bridge (forward decls only, no HIP runtime).
  target_compile_definitions(gtsam_points PRIVATE USE_HIP)
  target_compile_options(gtsam_points PRIVATE
    -include ${CMAKE_CURRENT_SOURCE_DIR}/include/gtsam_points/cuda/cuda_to_hip_types.h
  )
endif()

# GPU-related
if(BUILD_WITH_CUDA OR BUILD_WITH_HIP)
  # On Windows the GPU lib and the main lib have a mutual symbol dependency
  # (PointCloudGPU inherits PointCloudCPU across the library boundary).
  # Build as STATIC on Windows/HIP so the GPU objects are absorbed into
  # gtsam_points.dll at final link, avoiding the circular DLL import.
  # On Linux/CUDA the existing SHARED build is preserved.
  if(WIN32 AND BUILD_WITH_HIP)
    set(_gtsam_points_cuda_type STATIC)
  else()
    set(_gtsam_points_cuda_type SHARED)
  endif()
  add_library(gtsam_points_cuda ${_gtsam_points_cuda_type}
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

  if(BUILD_WITH_HIP)
    # Compile the .cu sources as HIP. The .cpp sources here are host-only
    # orchestration (streams + async alloc, no device kernels), so they stay
    # CXX; both languages get the compat header force-included and the
    # CUDA-toolkit-named forwarding shims on a PRIVATE+BEFORE include path.
    set(gtsam_points_cuda_hip_sources
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
      src/gtsam_points/cuda/stream_roundrobin.cu
      src/gtsam_points/cuda/stream_temp_buffer_roundrobin.cu
      src/gtsam_points/types/point_cloud.cu
      src/gtsam_points/types/point_cloud_gpu.cu
      src/gtsam_points/types/gaussian_voxelmap_gpu.cu
      src/gtsam_points/types/gaussian_voxelmap_gpu_funcs.cu
      src/gtsam_points/factors/integrated_vgicp_derivatives.cu
      src/gtsam_points/factors/integrated_vgicp_derivatives_inliers.cu
      src/gtsam_points/factors/integrated_vgicp_derivatives_compute.cu
      src/gtsam_points/factors/integrated_vgicp_derivatives_linearize.cu
      src/gtsam_points/util/easy_profiler_cuda.cu
    )
    set_source_files_properties(${gtsam_points_cuda_hip_sources} PROPERTIES LANGUAGE HIP)
    set_target_properties(gtsam_points_cuda PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

    # PRIVATE+BEFORE so <cuda.h>/<cuda_runtime.h>/<device_atomic_functions.h>/
    # <curand.h>/<cusparse.h> resolve to the forwarding shims on the HIP build
    # only (the dir is absent on the CUDA build, so the real toolkit wins).
    target_include_directories(gtsam_points_cuda BEFORE PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat
    )
    # Force-include the compat header into every TU (CXX and HIP) so sources
    # keep their CUDA spelling and the CUstream_st/graph type bridge is in scope.
    target_compile_options(gtsam_points_cuda PRIVATE
      -include ${CMAKE_CURRENT_SOURCE_DIR}/include/gtsam_points/cuda/cuda_to_hip.h
    )
    target_compile_definitions(gtsam_points_cuda PRIVATE USE_HIP)
  endif()

  if(BUILD_WITH_CUDA)
    target_link_libraries(gtsam_points_cuda CUDA::cudart)
  endif()
  if(BUILD_WITH_HIP)
    # hip::host (runtime) only -- NOT hip::device, whose -x hip/--offload-arch in
    # INTERFACE_COMPILE_OPTIONS would propagate to the host .cpp in this target.
    target_link_libraries(gtsam_points_cuda hip::host)
  endif()
  target_link_libraries(gtsam_points_cuda
    Boost::boost
    Eigen3::Eigen
    gtsam
    gtsam_unstable
    $<TARGET_NAME_IF_EXISTS:TBB::tbb>
    $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
  )
  if(NOT WIN32 OR NOT BUILD_WITH_HIP)
    set_target_properties(gtsam_points_cuda PROPERTIES
      VERSION   ${PROJECT_VERSION}
      SOVERSION 1
    )
  endif()

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
    if(BUILD_WITH_HIP)
      # Tests call the GPU API across CUstream_st*; force-include the HIP opaque-
      # type bridge so those calls match the gtsam_points_cuda definitions.
      target_compile_definitions(${test_name} PRIVATE USE_HIP)
      target_compile_options(${test_name} PRIVATE
        -include ${CMAKE_CURRENT_SOURCE_DIR}/include/gtsam_points/cuda/cuda_to_hip_types.h
      )
    endif()
    if(WIN32 AND BUILD_WITH_HIP)
      # The mixed MSVC/clang toolchain causes duplicate COMDAT symbols when
      # clang-compiled gtsam_points.lib and MSVC-compiled gtsam.dll both define
      # the same STL inline/template symbols. Use /FORCE:MULTIPLE to let lld-link
      # pick one definition and proceed (all definitions are functionally identical).
      target_link_options(${test_name} PRIVATE "LINKER:/FORCE:MULTIPLE")
    endif()
    if(WIN32 AND BUILD_WITH_HIP)
      # Defer test discovery to run time: build-time discovery requires the exe
      # to load successfully, but gtsam.dll and ROCm DLLs are not in the exe dir
      # until the DLL-copy step runs at ctest time.
      gtest_discover_tests(${test_name} WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        DISCOVERY_MODE PRE_TEST)
    else()
      gtest_discover_tests(${test_name} WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    endif()
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
if(BUILD_WITH_CUDA OR BUILD_WITH_HIP)
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
