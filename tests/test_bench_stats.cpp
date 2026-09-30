// CPU-only unit test for Phase 1 BenchStats percentiles.
#include "bench_stats.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>

static int g_fail = 0;

static void expect_near(const char* label, double got, double want, double eps = 1e-9) {
    if (std::fabs(got - want) > eps) {
        std::fprintf(stderr, "FAILED: %s got=%.9f want=%.9f\n", label, got, want);
        g_fail++;
    } else {
        std::printf("passed: %s\n", label);
    }
}

int main() {
    BenchStats s;
    // Odd count: exact middle for p50
    for (double v : {1.0, 2.0, 3.0, 4.0, 100.0}) s.add(v);
    expect_near("mean", s.mean(), 22.0);
    expect_near("p50", s.p50(), 3.0);
    expect_near("min", s.min(), 1.0);
    expect_near("max", s.max(), 100.0);
    // p95/p99 interpolate toward the high end on this tiny sample
    if (!(s.p95() >= s.p50() && s.p99() >= s.p95())) {
        std::fprintf(stderr, "FAILED: percentile ordering\n");
        g_fail++;
    } else {
        std::printf("passed: percentile ordering\n");
    }

    try {
        BenchStats empty;
        empty.mean();
        std::fprintf(stderr, "FAILED: empty mean should throw\n");
        g_fail++;
    } catch (...) {
        std::printf("passed: empty stats throws\n");
    }

    return g_fail == 0 ? 0 : 1;
}
