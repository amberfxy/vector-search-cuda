"""
End-to-end demo: use the custom CUDA kernel to rerank a corpus against a
query, the way it would actually be used in a retrieval pipeline (as
opposed to benchmark_torch.py, which only measures the isolated distance
computation).

This uses randomly generated stand-in vectors, NOT real sentence
embeddings -- this sandbox has no network access to download a real
embedding model (e.g. sentence-transformers). To make this a fully
realistic RAG-pipeline demo, swap `torch.randn(...)` below for real
embeddings, e.g.:

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    store = torch.from_numpy(model.encode(documents)).cuda()
    query = torch.from_numpy(model.encode([query_text])).cuda().squeeze(0)

which would tie this directly back to the Financial Market Intelligence
RAG System's actual embedding pipeline.

Run with: python demo_rerank.py
"""
import torch
import vector_search_torch as vst


def main():
    if not torch.cuda.is_available():
        print("No CUDA device available -- this demo requires a GPU.")
        return

    torch.manual_seed(0)
    num_documents = 100_000
    dim = 384

    print(f"Generating {num_documents} stand-in document embeddings (dim={dim})...")
    document_store = torch.randn(num_documents, dim, device="cuda", dtype=torch.float32)
    query_embedding = torch.randn(dim, device="cuda", dtype=torch.float32)

    k = 5
    top_indices, top_scores = vst.rerank(query_embedding, document_store, k=k, metric="cosine")

    print(f"\nTop-{k} most similar documents (custom CUDA kernel + torch.topk):")
    for rank, (idx, score) in enumerate(zip(top_indices.tolist(), top_scores.tolist()), start=1):
        print(f"  #{rank}  doc_index={idx:<8}  cosine_similarity={score:.4f}")

    print("\nThis is the shape of a real retrieval step: embed a query, score it "
          "against a corpus with the custom kernel, take the top-k. Swap in real "
          "sentence embeddings (see module docstring) to connect this directly "
          "to the Financial Market Intelligence RAG System's actual data.")


if __name__ == "__main__":
    main()
