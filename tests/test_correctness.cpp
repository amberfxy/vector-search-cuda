// Lightweight correctness tests -- no external test framework dependency,
// so this compiles anywhere with just a C++17 compiler.
//
// What this verifies:
//   1. Single-threaded and OpenMP batch distance results match (they must,
//      since correctness shouldn't depend on thread count).
//   2. L2 distance and cosine similarity match hand-computed values on a
//      tiny known example.
//   3. CPU top-k selection returns the correct indices in the correct order.
//
// This is the ground truth the CUDA kernels get checked against once you
// build this on a machine with a GPU: run the same query/store through the
// naive and tiled CUDA kernels and diff the output against
// batch_distance_singlethread's output (tolerance ~1e-4 for float rounding).
#include "vector_store.hpp"
#include "distance_cpu.hpp"
#include "topk_cpu.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>

static int g_failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAILED: %s (line %d)\n", msg, __LINE__); \
        g_failures++; \
    } else { \
        std::printf("passed: %s\n", msg); \
    } \
} while (0)

static bool approx_eq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

void test_known_values() {
    float a[3] = {1.0f, 0.0f, 0.0f};
    float b[3] = {0.0f, 1.0f, 0.0f};
    // L2 distance between orthogonal unit vectors is sqrt(2).
    CHECK(approx_eq(l2_distance(a, b, 3), std::sqrt(2.0f)), "L2 distance of orthogonal unit vectors == sqrt(2)");
    // Cosine similarity between orthogonal vectors is 0.
    CHECK(approx_eq(cosine_similarity(a, b, 3), 0.0f), "cosine similarity of orthogonal vectors == 0");

    float c[3] = {2.0f, 0.0f, 0.0f};
    // Cosine similarity between parallel (same-direction) vectors is 1, regardless of magnitude.
    CHECK(approx_eq(cosine_similarity(a, c, 3), 1.0f), "cosine similarity of parallel vectors == 1");
}

void test_singlethread_matches_openmp() {
    const size_t n = 5000, dim = 128;
    VectorStore store(n, dim);
    store.fillRandom(/*seed=*/7, /*normalize=*/true);

    std::vector<float> query(dim);
    {
        VectorStore qstore(1, dim);
        qstore.fillRandom(/*seed=*/99, true);
        std::copy(qstore.raw(), qstore.raw() + dim, query.begin());
    }

    std::vector<float> scores_st(n), scores_omp(n);
    batch_distance_singlethread(store, query.data(), Metric::L2, scores_st.data());
    batch_distance_openmp(store, query.data(), Metric::L2, scores_omp.data());

    bool all_match = true;
    for (size_t i = 0; i < n; ++i) {
        if (!approx_eq(scores_st[i], scores_omp[i])) { all_match = false; break; }
    }
    CHECK(all_match, "single-threaded and OpenMP L2 batch results match");

    batch_distance_singlethread(store, query.data(), Metric::Cosine, scores_st.data());
    batch_distance_openmp(store, query.data(), Metric::Cosine, scores_omp.data());
    all_match = true;
    for (size_t i = 0; i < n; ++i) {
        if (!approx_eq(scores_st[i], scores_omp[i])) { all_match = false; break; }
    }
    CHECK(all_match, "single-threaded and OpenMP cosine batch results match");
}

void test_topk_smallest() {
    float scores[8] = {5.0f, 1.0f, 9.0f, 3.0f, 7.0f, 0.5f, 2.0f, 8.0f};
    auto top3 = top_k_smallest(scores, 8, 3);
    CHECK(top3.size() == 3, "top_k_smallest returns k results");
    // Expected order (ascending): index 5 (0.5), index 1 (1.0), index 3 (3.0)
    CHECK(top3[0].index == 5 && approx_eq(top3[0].score, 0.5f), "top_k_smallest rank 0 correct");
    CHECK(top3[1].index == 1 && approx_eq(top3[1].score, 1.0f), "top_k_smallest rank 1 correct");
    CHECK(top3[2].index == 6 && approx_eq(top3[2].score, 2.0f), "top_k_smallest rank 2 correct");
}

int main() {
    test_known_values();
    test_singlethread_matches_openmp();
    test_topk_smallest();

    if (g_failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    } else {
        std::printf("\n%d test(s) FAILED.\n", g_failures);
        return 1;
    }
}
