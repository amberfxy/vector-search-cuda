#pragma once
#include "vector_store.hpp"
#include <fstream>
#include <stdexcept>

// Minimal raw binary format so you can also dump real embeddings from your
// Python RAG project (numpy float32 array, shape [N, dim]) and load them
// here instead of random data -- makes the benchmark reflect your actual
// financial-news embedding distribution, not just synthetic Gaussian noise.
//
// Format: no header, just num_vectors * dim contiguous float32 values,
// row-major -- i.e. exactly `arr.astype(np.float32).tofile(path)` from numpy.
inline void save_raw(const VectorStore& store, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Could not open " + path + " for writing");
    out.write(reinterpret_cast<const char*>(store.raw()), store.sizeBytes());
}

inline VectorStore load_raw(const std::string& path, size_t num_vectors, size_t dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Could not open " + path + " for reading");
    VectorStore store(num_vectors, dim);
    in.read(reinterpret_cast<char*>(store.raw()), store.sizeBytes());
    if (!in) throw std::runtime_error("File " + path + " shorter than expected " +
                                       "(check num_vectors/dim match the file)");
    return store;
}
