cmake_minimum_required(VERSION 3.24)
project(MiniLLMRT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

option(MINILLM_WITH_TENSORRT "Build the TensorRT backend" ON)
option(MINILLM_BUILD_TESTS "Build unit tests" ON)

find_package(CUDAToolkit REQUIRED)

add_library(minillm_core
    src/kv_block_manager.cpp
    src/scheduler.cpp
    src/runtime_session.cpp
    src/greedy_sampler.cpp)
target_include_directories(minillm_core PUBLIC include)

add_library(minillm_cuda cuda/kv_cache_kernels.cu)
target_include_directories(minillm_cuda PUBLIC include)
target_link_libraries(minillm_cuda PUBLIC CUDA::cudart)
target_compile_options(minillm_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

if(MINILLM_WITH_TENSORRT)
  set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT SDK root")
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
      HINTS ${TENSORRT_ROOT}/include ${CUDAToolkit_INCLUDE_DIRS})
  find_library(TENSORRT_LIBRARY nvinfer
      HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64)
  if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT C++ SDK not found; set TENSORRT_ROOT")
  endif()
  add_library(minillm_trt src/tensorrt_backend.cpp)
  target_include_directories(minillm_trt PUBLIC include ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(minillm_trt PUBLIC CUDA::cudart ${TENSORRT_LIBRARY})
  target_compile_definitions(minillm_trt PUBLIC MINILLM_WITH_TENSORRT=1)
endif()

add_executable(minillm_cli src/main.cpp)
target_link_libraries(minillm_cli PRIVATE minillm_core minillm_cuda)
if(TARGET minillm_trt)
  target_link_libraries(minillm_cli PRIVATE minillm_trt)
endif()

if(MINILLM_BUILD_TESTS)
  enable_testing()
  add_executable(test_kv_block_manager tests/test_kv_block_manager.cpp)
  target_link_libraries(test_kv_block_manager PRIVATE minillm_core)
  add_test(NAME kv_block_manager COMMAND test_kv_block_manager)

  add_executable(test_scheduler tests/test_scheduler.cpp)
  target_link_libraries(test_scheduler PRIVATE minillm_core)
  add_test(NAME scheduler COMMAND test_scheduler)

  add_executable(test_scheduler_stress tests/test_scheduler_stress.cpp)
  target_link_libraries(test_scheduler_stress PRIVATE minillm_core)
  add_test(NAME scheduler_stress COMMAND test_scheduler_stress)

  add_executable(test_kv_cache_cuda tests/test_kv_cache_cuda.cu)
  target_link_libraries(test_kv_cache_cuda PRIVATE minillm_cuda CUDA::cudart)
  target_compile_options(test_kv_cache_cuda PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  add_test(NAME kv_cache_cuda COMMAND test_kv_cache_cuda)

  add_executable(benchmark_kv_cache_cuda benchmarks/benchmark_kv_cache_cuda.cu)
  target_link_libraries(benchmark_kv_cache_cuda PRIVATE minillm_cuda CUDA::cudart)
endif()
