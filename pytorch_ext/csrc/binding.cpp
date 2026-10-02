// PyTorch-facing binding layer: tensor validation + dispatch.
#include <torch/extension.h>

namespace vector_search_torch {
void launch_l2_tiled(const float* store, const float* query,
                      int64_t num_vectors, int64_t dim, float* out_scores);
void launch_cosine_tiled(const float* store, const float* query,
                          int64_t num_vectors, int64_t dim, float* out_scores);
void launch_l2_warp(const float* store, const float* query,
                    int64_t num_vectors, int64_t dim, float* out_scores);
void launch_l2_tiled_batch(const float* store, const float* queries,
                            int64_t num_vectors, int64_t dim, int64_t num_queries,
                            float* out_scores);
void launch_l2_warp_batch(const float* store, const float* queries,
                           int64_t num_vectors, int64_t dim, int64_t num_queries,
                           float* out_scores);
}

namespace {

void check_pair(const torch::Tensor& query, const torch::Tensor& store) {
    TORCH_CHECK(query.is_cuda() && store.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(query.dtype() == torch::kFloat32 && store.dtype() == torch::kFloat32,
                "float32 required");
    TORCH_CHECK(query.dim() == 1, "query must be 1D [dim]");
    TORCH_CHECK(store.dim() == 2, "store must be 2D [N, dim]");
    TORCH_CHECK(query.size(0) == store.size(1), "dim mismatch");
    TORCH_CHECK(query.is_contiguous() && store.is_contiguous(), "must be contiguous");
}

void check_batch(const torch::Tensor& queries, const torch::Tensor& store) {
    TORCH_CHECK(queries.is_cuda() && store.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(queries.dtype() == torch::kFloat32 && store.dtype() == torch::kFloat32,
                "float32 required");
    TORCH_CHECK(queries.dim() == 2, "queries must be 2D [Q, dim]");
    TORCH_CHECK(store.dim() == 2, "store must be 2D [N, dim]");
    TORCH_CHECK(queries.size(1) == store.size(1), "dim mismatch");
    TORCH_CHECK(queries.is_contiguous() && store.is_contiguous(), "must be contiguous");
}

} // namespace

torch::Tensor tiled_l2_distance(torch::Tensor query, torch::Tensor store) {
    check_pair(query, store);
    auto out = torch::empty({store.size(0)}, query.options());
    vector_search_torch::launch_l2_tiled(
        store.data_ptr<float>(), query.data_ptr<float>(),
        store.size(0), store.size(1), out.data_ptr<float>());
    return out;
}

torch::Tensor tiled_cosine_similarity(torch::Tensor query, torch::Tensor store) {
    check_pair(query, store);
    auto out = torch::empty({store.size(0)}, query.options());
    vector_search_torch::launch_cosine_tiled(
        store.data_ptr<float>(), query.data_ptr<float>(),
        store.size(0), store.size(1), out.data_ptr<float>());
    return out;
}

torch::Tensor warp_l2_distance(torch::Tensor query, torch::Tensor store) {
    check_pair(query, store);
    auto out = torch::empty({store.size(0)}, query.options());
    vector_search_torch::launch_l2_warp(
        store.data_ptr<float>(), query.data_ptr<float>(),
        store.size(0), store.size(1), out.data_ptr<float>());
    return out;
}

torch::Tensor tiled_l2_distance_batch(torch::Tensor queries, torch::Tensor store) {
    check_batch(queries, store);
    auto out = torch::empty({queries.size(0), store.size(0)}, queries.options());
    vector_search_torch::launch_l2_tiled_batch(
        store.data_ptr<float>(), queries.data_ptr<float>(),
        store.size(0), store.size(1), queries.size(0), out.data_ptr<float>());
    return out;
}

torch::Tensor warp_l2_distance_batch(torch::Tensor queries, torch::Tensor store) {
    check_batch(queries, store);
    auto out = torch::empty({queries.size(0), store.size(0)}, queries.options());
    vector_search_torch::launch_l2_warp_batch(
        store.data_ptr<float>(), queries.data_ptr<float>(),
        store.size(0), store.size(1), queries.size(0), out.data_ptr<float>());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiled_l2_distance", &tiled_l2_distance,
          "Tiled L2: query [dim] vs store [N,dim] -> [N]");
    m.def("tiled_cosine_similarity", &tiled_cosine_similarity,
          "Tiled cosine: query [dim] vs store [N,dim] -> [N]");
    m.def("warp_l2_distance", &warp_l2_distance,
          "Warp-per-vector L2: query [dim] vs store [N,dim] -> [N]");
    m.def("tiled_l2_distance_batch", &tiled_l2_distance_batch,
          "Batched tiled L2: queries [Q,dim] vs store [N,dim] -> [Q,N]");
    m.def("warp_l2_distance_batch", &warp_l2_distance_batch,
          "Batched warp L2: queries [Q,dim] vs store [N,dim] -> [Q,N]");
}
