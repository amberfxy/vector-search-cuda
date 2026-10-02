#pragma once
// Consistent CUDA error checking for the library (host code).
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace vsc {

inline std::string cuda_err_string(cudaError_t err, const char* expr,
                                   const char* file, int line) {
    return std::string("CUDA error: ") + cudaGetErrorString(err) +
           " (" + expr + ") at " + file + ":" + std::to_string(line);
}

[[noreturn]] inline void throw_cuda(cudaError_t err, const char* expr,
                                    const char* file, int line) {
    if (err == cudaErrorMemoryAllocation) {
        throw std::runtime_error(
            cuda_err_string(err, expr, file, line) +
            " — GPU out of memory. Reduce N/dim/batch or free other device memory.");
    }
    if (err == cudaErrorInvalidDevice) {
        throw std::runtime_error(
            cuda_err_string(err, expr, file, line) +
            " — invalid CUDA device. Check cudaGetDeviceCount / --device.");
    }
    throw std::runtime_error(cuda_err_string(err, expr, file, line));
}

} // namespace vsc

#define VSC_CUDA_CHECK(call) do { \
    cudaError_t _vsc_err = (call); \
    if (_vsc_err != cudaSuccess) { \
        ::vsc::throw_cuda(_vsc_err, #call, __FILE__, __LINE__); \
    } \
} while (0)

#define VSC_CUDA_CHECK_LAUNCH() do { \
    cudaError_t _vsc_err = cudaGetLastError(); \
    if (_vsc_err != cudaSuccess) { \
        ::vsc::throw_cuda(_vsc_err, "kernel launch", __FILE__, __LINE__); \
    } \
} while (0)
