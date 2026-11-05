cmake_minimum_required(VERSION 3.18)
# Configure (nvcc is not on PATH; ROS must be sourced in bash only if building the bench):
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_CUDA_COMPILER=$HOME/miniforge3/envs/cuda128/bin/nvcc
project(gpu_frontend LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Jetson Orin (sm_87) and Blackwell RTX 50 series / laptop 5060 (sm_120).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "87;120")
endif()

set(UWCT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../uw_ct_vio" CACHE PATH "uw_ct_vio checkout (read only)")
set(GTSAM_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../deps/gtsam-4.2.0" CACHE PATH "GTSAM 4.2 prefix")
set(GTSAM_DIR "${GTSAM_PREFIX}/lib/cmake/GTSAM" CACHE PATH "")

find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs video calib3d)
find_package(CUDAToolkit REQUIRED)

# CUDA kernels: no OpenCV/GTSAM headers reach nvcc.
add_library(gpu_frontend_core STATIC src/GpuKltCore.cu)
target_include_directories(gpu_frontend_core PUBLIC include)
set_target_properties(gpu_frontend_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)
target_link_libraries(gpu_frontend_core PUBLIC CUDA::cudart)

# CPU reference: the unmodified uw_ct_vio tracker, compiled from its source path.
add_library(uwct_klt_ref STATIC ${UWCT_DIR}/src/KltStereoTracker.cpp)
target_include_directories(uwct_klt_ref PUBLIC ${UWCT_DIR}/include)
target_link_libraries(uwct_klt_ref PUBLIC gtsam ${OpenCV_LIBS})
target_compile_options(uwct_klt_ref PRIVATE -O3 -w)

add_library(gpu_frontend STATIC src/GpuKltStereoTracker.cpp)
target_include_directories(gpu_frontend PUBLIC include)
target_compile_options(gpu_frontend PRIVATE -Wall -Wextra)
target_link_libraries(gpu_frontend PUBLIC gpu_frontend_core uwct_klt_ref)

enable_testing()
find_package(GTest QUIET)
if(GTest_FOUND)
  add_executable(test_gpu_frontend tests/test_gpu_frontend.cpp)
  target_link_libraries(test_gpu_frontend gpu_frontend GTest::gtest GTest::gtest_main)
  set_target_properties(test_gpu_frontend PROPERTIES BUILD_RPATH "${GTSAM_PREFIX}/lib")
  add_test(NAME test_gpu_frontend COMMAND test_gpu_frontend)
endif()

# Bench needs ROS 2 (source /opt/ros/jazzy/setup.bash in bash before configuring).
find_package(rclcpp QUIET)
find_package(rosbag2_cpp QUIET)
find_package(sensor_msgs QUIET)
find_package(yaml-cpp QUIET)
if(rclcpp_FOUND AND rosbag2_cpp_FOUND AND sensor_msgs_FOUND AND yaml-cpp_FOUND)
  add_executable(bench_frontend bench/bench_frontend.cpp)
  target_link_libraries(bench_frontend gpu_frontend yaml-cpp rclcpp::rclcpp rosbag2_cpp::rosbag2_cpp
                        ${sensor_msgs_TARGETS})
  set_target_properties(bench_frontend PROPERTIES BUILD_RPATH "${GTSAM_PREFIX}/lib")
else()
  message(STATUS "ROS 2 not found: bench_frontend not built")
endif()
