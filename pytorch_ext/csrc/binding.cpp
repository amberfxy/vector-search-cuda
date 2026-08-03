// PyTorch-facing binding layer. This is where tensor validation happens
// (device, dtype, contiguity, shape) -- keeping it separate from
// tiled_kernels.cu means the CUDA file only ever deals with raw pointers
// it can already trust are correct, and this file only ever deals with
// torch::Tensor and doesn't know about CUDA launch configuration.
#include <torch/extension.h>

namespace vector_search_torch {
// Declared in tiled_kernels.cu
void launch_l2_tiled(const float* store, const float* query,
                      int64_t num_vectors, int64_t dim, float* out_scores);
void launch_cosine_tiled(const float* store, const float* query,
                          int64_t num_vectors, int64_t dim, float* out_scores);
}

namespace {

void check_inputs(const torch::Tensor& query, const torch::Tensor& store) {
    TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor (call .cuda() on it first)");
    TORCH_CHECK(store.is_cuda(), "store must be a CUDA tensor (call .cuda() on it first)");
    TORCH_CHECK(query.dtype() == torch::kFloat32, "query must be float32");
    TORCH_CHECK(store.dtype() == torch::kFloat32, "store must be float32");
    TORCH_CHECK(query.dim() == 1, "query must be 1D, shape [dim]");
    TORCH_CHECK(store.dim() == 2, "store must be 2D, shape [num_vectors, dim]");
    TORCH_CHECK(query.size(0) == store.size(1),
                "query dim (", query.size(0), ") must match store's second dimension (", store.size(1), ")");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous (call .contiguous() first)");
    TORCH_CHECK(store.is_contiguous(), "store must be contiguous (call .contiguous() first)");
}

} // namespace

// tiled_l2_distance(query, store) -> Tensor[num_vectors]
// Returns L2 distance from `query` to every row of `store`. Smaller = closer.
torch::Tensor tiled_l2_distance(torch::Tensor query, torch::Tensor store) {
    check_inputs(query, store);
    const int64_t num_vectors = store.size(0);
    const int64_t dim = store.size(1);

    auto out = torch::empty({num_vectors}, query.options());
    vector_search_torch::launch_l2_tiled(
        store.data_ptr<float>(), query.data_ptr<float>(),
        num_vectors, dim, out.data_ptr<float>());

    // Kernel launches are asynchronous; PyTorch's caching allocator and
    // stream semantics handle this correctly for normal use (the next op
    // that reads `out` on the same stream will wait for it), so no explicit
    // cudaDeviceSynchronize() here. For benchmarking, wrap calls with
    // torch.cuda.synchronize() / torch.cuda.Event, as benchmark_torch.py does.
    return out;
}

// tiled_cosine_similarity(query, store) -> Tensor[num_vectors]
// Returns cosine similarity from `query` to every row of `store`. Larger = more similar.
torch::Tensor tiled_cosine_similarity(torch::Tensor query, torch::Tensor store) {
    check_inputs(query, store);
    const int64_t num_vectors = store.size(0);
    const int64_t dim = store.size(1);

    auto out = torch::empty({num_vectors}, query.options());
    vector_search_torch::launch_cosine_tiled(
        store.data_ptr<float>(), query.data_ptr<float>(),
        num_vectors, dim, out.data_ptr<float>());

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiled_l2_distance", &tiled_l2_distance,
          "Shared-memory-tiled CUDA L2 distance: query [dim] vs store [N, dim] -> [N]");
    m.def("tiled_cosine_similarity", &tiled_cosine_similarity,
          "Shared-memory-tiled CUDA cosine similarity: query [dim] vs store [N, dim] -> [N]");
}
