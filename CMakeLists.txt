cmake_minimum_required(VERSION 3.18)

# CUDA must be enabled as a language BEFORE project() if we want it available;
# but we don't know yet whether nvcc exists on this machine. The pattern
# below (check_language) lets this same CMakeLists.txt build a CPU-only
# version on a laptop with no GPU, and the full GPU version on a CUDA machine,
# without editing anything.
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    project(vector_search_cuda LANGUAGES CXX CUDA)
    set(HAVE_CUDA TRUE)
    message(STATUS "CUDA compiler found: building GPU kernels")
else()
    project(vector_search_cuda LANGUAGES CXX)
    set(HAVE_CUDA FALSE)
    message(WARNING "No CUDA compiler found -- building CPU-only. "
                     "Install the CUDA toolkit to build the GPU kernels.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

include_directories(include)

# --- CPU library (always built) ---
add_library(vector_search_cpu STATIC
    src/cpu/distance_cpu.cpp
    src/cpu/topk_cpu.cpp
)
target_link_libraries(vector_search_cpu PUBLIC OpenMP::OpenMP_CXX)

# --- CUDA library (only if a CUDA compiler was found) ---
if(HAVE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)  # covers Volta through Ampere; add 89/90 for Ada/Hopper if needed

    add_library(vector_search_cuda STATIC
        src/cuda/cuda_timing.cu
        src/cuda/distance_naive.cu
        src/cuda/distance_tiled.cu
        src/cuda/topk_cuda.cu
    )
    set_target_properties(vector_search_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# --- Benchmark executable ---
add_executable(bench_runner src/benchmark/bench_runner.cpp)
target_link_libraries(bench_runner PRIVATE vector_search_cpu)
if(HAVE_CUDA)
    target_compile_definitions(bench_runner PRIVATE USE_CUDA)
    target_link_libraries(bench_runner PRIVATE vector_search_cuda)
endif()

# --- Correctness tests ---
enable_testing()
add_executable(test_correctness tests/test_correctness.cpp)
target_link_libraries(test_correctness PRIVATE vector_search_cpu)
add_test(NAME correctness COMMAND test_correctness)

if(HAVE_CUDA)
    add_executable(test_correctness_gpu tests/test_correctness_gpu.cpp)
    target_link_libraries(test_correctness_gpu PRIVATE vector_search_cpu vector_search_cuda)
    add_test(NAME correctness_gpu COMMAND test_correctness_gpu)
    message(STATUS "GPU correctness test enabled: run 'ctest' or './test_correctness_gpu' after building")
endif()
