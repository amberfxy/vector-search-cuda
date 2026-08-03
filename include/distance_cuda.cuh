#pragma once
#include <cstddef>
#include "distance_cpu.hpp"  // reuse the Metric enum

// Both functions share the same contract as the CPU batch functions:
// compute distance/similarity from one query vector against every vector
// in a flat, row-major device buffer of `num_vectors` vectors of `dim`
// floats each, writing `num_vectors` results into out_scores.
//
// These are HOST functions -- they own the cudaMalloc/cudaMemcpy/kernel
// launch/cudaMemcpy-back/cudaFree lifecycle internally, so bench_runner.cpp
// can call them exactly like the CPU functions and only needs host-side
// pointers. (In a production system you'd keep data resident on the device
// across many queries rather than re-transferring every call -- see the
// README "Known limitations" section for why this benchmark intentionally
// keeps that transfer in the timed region.)

void batch_distance_naive_cuda(const float* h_store, size_t num_vectors, size_t dim,
                                const float* h_query, Metric metric,
                                float* h_out_scores);

void batch_distance_tiled_cuda(const float* h_store, size_t num_vectors, size_t dim,
                                const float* h_query, Metric metric,
                                float* h_out_scores);

// Returns elapsed GPU compute time in milliseconds for the most recent call
// (measured with cudaEvent, kernel-launch-to-completion only, excluding
// host<->device transfer). Benchmark code calls this right after each
// batch_distance_*_cuda call to separate "transfer time" from "compute time"
// in the results -- see README for why that split matters.
float last_kernel_time_ms();
