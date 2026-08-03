#pragma once
#include <vector>
#include <random>
#include <stdexcept>
#include <cstring>

// VectorStore holds N vectors of dimension D in a single contiguous buffer,
// row-major layout: vector i occupies data[i*dim ... i*dim + dim - 1].
//
// Design note (worth being able to explain in an interview):
// A naive design would use std::vector<std::vector<float>>, which scatters
// each vector across a separate heap allocation. That kills CPU cache
// locality for batch distance computation, and it's actively hostile to GPU
// transfer: you'd need N separate cudaMemcpy calls (or an array of device
// pointers) instead of one contiguous cudaMemcpy. A single flat buffer with
// row-major layout is the standard choice for anything that will eventually
// touch a GPU or a BLAS-style routine (this is exactly how FAISS, cuBLAS,
// and friends lay out their data internally).
class VectorStore {
public:
    VectorStore(size_t num_vectors, size_t dim)
        : num_vectors_(num_vectors), dim_(dim), data_(num_vectors * dim) {}

    // Fill with reproducible pseudo-random data (seeded), optionally
    // L2-normalized so cosine similarity and L2 distance are both meaningful
    // on the same dataset.
    void fillRandom(unsigned seed = 42, bool normalize = true) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < num_vectors_; ++i) {
            float* v = vectorAt(i);
            float norm_sq = 0.0f;
            for (size_t d = 0; d < dim_; ++d) {
                v[d] = dist(rng);
                norm_sq += v[d] * v[d];
            }
            if (normalize && norm_sq > 0.0f) {
                float inv_norm = 1.0f / std::sqrt(norm_sq);
                for (size_t d = 0; d < dim_; ++d) v[d] *= inv_norm;
            }
        }
    }

    float* vectorAt(size_t i) {
        if (i >= num_vectors_) throw std::out_of_range("VectorStore index out of range");
        return data_.data() + i * dim_;
    }
    const float* vectorAt(size_t i) const {
        if (i >= num_vectors_) throw std::out_of_range("VectorStore index out of range");
        return data_.data() + i * dim_;
    }

    float* raw() { return data_.data(); }
    const float* raw() const { return data_.data(); }

    size_t numVectors() const { return num_vectors_; }
    size_t dim() const { return dim_; }
    size_t sizeBytes() const { return data_.size() * sizeof(float); }

private:
    size_t num_vectors_;
    size_t dim_;
    std::vector<float> data_;
};
