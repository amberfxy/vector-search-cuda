#include "cuda_timing.cuh"
#include "distance_cuda.cuh"

namespace {
float g_last_kernel_ms = 0.0f;
}

void set_last_kernel_time_ms(float ms) { g_last_kernel_ms = ms; }
float last_kernel_time_ms() { return g_last_kernel_ms; }
