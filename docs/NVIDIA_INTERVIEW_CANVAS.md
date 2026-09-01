# NVIDIA Interview Canvas — vector-search-cuda + Financial RAG

> **Purpose:** One scrollable review panel for NVIDIA-style GPU / systems interviews.  
> **Projects:** [`vector-search-cuda`](https://github.com/amberfxy/vector-search-cuda) (this repo) + companion [`financial-market-intelligence-rag`](https://github.com/amberfxy/financial-market-intelligence-rag).  
> **Verified hardware:** Google Colab **NVIDIA Tesla T4** (15 GB), nvcc **12.8**, 50 queries/size (dim=384 primary table).

---

## 0. Elevator pitch (30 sec)

> I built a **Financial Market Intelligence RAG** system that retrieves with **FAISS CPU `IndexFlatL2`** and BGE-large embeddings. Separately, **`vector-search-cuda`** asks: what does it take to implement the same **exact brute-force L2 search** on GPU by hand? I started from a **correct naive kernel**, applied **shared-memory query staging**, measured honestly against CPU/OpenMP/FAISS, and extended it as a **PyTorch CUDA op** on device-resident tensors. The story is not “I beat FAISS everywhere” — it’s **where speedups come from**, **what Roofline predicts**, and **what I’d optimize next**.

---

## 1. Two-project map (how they connect)

| | **Financial RAG** (production-shaped) | **vector-search-cuda** (GPU lab) |
|---|---|---|
| Role | End-to-end retrieval + LLM + UI | Low-level distance kernel engineering |
| Vector search | FAISS CPU `IndexFlatL2` | Hand-written CUDA L2/cosine |
| Embedding | BGE-large, dim=**1024** | Benchmark at dim=**384** + **1024** |
| GPU in prod? | No (CPU FAISS path) | Yes (companion prototype) |
| Interview use | System design, trade-offs, RAG pipeline | Kernel opt, memory hierarchy, Roofline, PyTorch extension |

**One-liner link:** RAG owns the **product**; CUDA repo owns the **retrieval primitive** underneath it.

---

## 2. Resume bullets → evidence (don’t overclaim)

| Bullet theme | Safe claim | Evidence in repo |
|---|---|---|
| CUDA kernel from scratch | Naive → shared-query tiled | `src/cuda/distance_naive.cu`, `distance_tiled.cu` |
| Measured speedup | **10K tiled 1.6×** over naive; **~10–13×** GPU vs OpenMP at 1M (kernel-only) | `results/benchmark.csv` |
| Honest mixed results | **100K–1M naive ≈ tiled** — candidate traffic dominates | README Results |
| Transfer vs compute | Wall ~391 ms vs kernel ~28 ms at 1M | `bench_runner` wall + `cudaEvent` |
| PyTorch integration | pybind11 extension, device-resident | `pytorch_ext/` |
| vs production libs | Fair baseline: FAISS **IndexFlatL2** CPU only; not IVF/HNSW | `scripts/faiss_baseline.py` |
| Correctness first | GPU vs CPU tol **1e-4** before trusting benches | `tests/test_correctness_gpu.cpp` |

**Do NOT say:** beat all of FAISS; cuBLAS layout identity; Nsight profiling done; warm-up×3 / median p95; block-per-vector reduction shipped; Thrust = partial top-k.

---

## 3. Repo architecture (file → interview talking point)

```
include/vector_store.hpp       row-major flat buffer (FAISS/cuBLAS layout)
src/cpu/distance_cpu.cpp       CPU ground truth (L2, cosine)
src/cuda/distance_naive.cu     baseline: 1 thread / candidate
src/cuda/distance_tiled.cu     optimization: shared-query staging
src/cuda/cuda_timing.cu        shared last_kernel_time_ms (avoid duplicate symbols)
src/cuda/topk_cuda.cu          Thrust full sort (not hand bitonic)
src/benchmark/bench_runner.cpp wall + kernel timing, CSV
tests/test_correctness_gpu.cpp RUN FIRST on GPU
pytorch_ext/csrc/binding.cpp   torch::Tensor validation + pybind11
pytorch_ext/csrc/tiled_kernels.cu  no cudaMalloc — device-resident
```

---

## 4. Kernel optimization narrative (Lecture 8 checklist)

Use this when asked: *“Walk me through optimizing a kernel end-to-end.”*

| Step | Reduction lecture | **This project — done** | **Next (honest future work)** |
|---|---|---|---|
| 0 Baseline | Correct serial op | `l2_naive_kernel`, CPU tests | — |
| 1 Roofline | Pick lever | AI ≈ 0.75 → memory-bound | Nsight transaction efficiency |
| 2 Divergence | Contiguous active lanes | N/A (no tree yet) | Block-per-vector dim reduction |
| 3 Shared memory | Stage hot data | **Query → `__shared__`** | Partial sums in shared |
| 4 Multiblock | Atomics | N/A | If dim > one block |
| 5 Coarsening | Fewer tree steps | N/A | Each thread handles multiple `d` |
| FP subtlety | Non-associativity | Serial fp32 sum today | Tree order if parallelized |

**English closing line:**
> “Each L2 distance is a reduction over dimensions — but I shipped serial per-thread reduction and one clear shared-memory win on the query. Block-per-vector tree reduction is the planned coalescing fix.”

---

## 5. Roofline cheat sheet (L2, one candidate)

### 5.1 Work & traffic

| Quantity | Formula | dim=384 | dim=1024 |
|---|---|---|---|
| FLOPs/candidate | **3D + 1** (sub, mul, add, sqrt) | **1153** | **3073** |
| Useful bytes read | **4D** (candidate) + 4 (score write) | ~1.5 KB | ~4 KB |
| Arithmetic intensity | **≈ 3D / 4D = 0.75 FLOP/byte** | 0.75 | 0.75 |

### 5.2 T4 roof

| Spec source | FP32 peak | BW peak | Ridge (FLOP/byte) |
|---|---|---|---|
| Datasheet | 8.1 TFLOP/s | **300 GB/s** | **≈27** |
| Some NVIDIA pages | 8.1 TFLOP/s | 320 GB/s | ≈25.3 |

**0.75 ≪ 25–27 → memory-bound** (optimize bytes/access, not FLOPs).

### 5.3 Three precision levels (GPT corrections — memorize)

1. **FLOPs = 3D + 1**, not “≈3D” when giving numbers.  
2. **~54 GB/s** = **effective useful BW** back-calculated from `N·D·4 / t_kernel` — **not** Nsight DRAM throughput.  
3. **Memory-bound ≠ bandwidth saturated.** Achieved ~**15–20%** of peak → strided access, transaction efficiency, latency, cache still matter.

### 5.4 Back-of-envelope (1M × 384, kernel ~28.3 ms)

- Useful candidate read: 1M × 384 × 4 ≈ **1.47 GB**  
- Effective BW ≈ 1.47 / 0.0283 ≈ **52 GB/s** (~16% of 320 GB/s)  
- Achieved GFLOPS ≈ 1M × 1153 / 0.0283e9 ≈ **41 GFLOPS** (~0.5% of 8.1 TFLOPS)

---

## 6. CUDA concepts — used or not?

| Concept | Status | Where / note |
|---|---|---|
| **Thread hierarchy** | ✅ Used | `blockIdx`, `threadIdx`, 256 threads/block, 1 thread/candidate |
| **Memory hierarchy** | ✅ Basic | Global + shared; L2 discussed analytically |
| **Shared memory** | ✅ Core opt | `extern __shared__`, dynamic smem, `__syncthreads` |
| **Async execution** | ✅ Basic | Async `<<<>>>`, sync for timing / D2H |
| **Verification** | ✅ Strong | CPU ground truth, tol 1e-4, n=257, naive vs tiled |
| **Timing** | ✅ Strong | `cudaEvent` kernel-only + wall-clock split |
| **Coalescing** | ❌ Not tuned | Strided `store[i*D+d]` — **known bottleneck** |
| **Occupancy** | ⚠️ Implicit | Fixed 256/block; **not measured/tuned** |

### Access pattern (why coalescing hurts)

- Warp threads `i, i+1, …` at fixed `d` read addresses **D floats apart** → poor merge.  
- Future **block-per-vector**: threads cooperate along `d` → better coalescing.

### Shared-query staging (what we actually optimized)

```cpp
// Cooperative load once per block
for (d = threadIdx.x; d < dim; d += blockDim.x)
    shared_query[d] = query[d];
__syncthreads();
// Then read query from shared, candidate from global
```

---

## 7. Benchmark numbers (Colab T4, commit to memory)

### 7.1 dim=384, L2, **kernel-only**, 50 queries

| N | CPU ST | CPU OMP | GPU naive | GPU tiled | FAISS CPU FlatL2 |
|---|---|---|---|---|---|
| 10K | 4.85 ms | 2.57 ms | 0.28 ms | **0.18 ms** | 4.30 ms |
| 100K | 50.8 ms | 37.0 ms | 2.68 ms | 2.78 ms | 45.0 ms |
| 1M | 486 ms | 286 ms | 28.3 ms | 28.4 ms | 416 ms |

**Patterns:**
- Tiled vs naive: **1.6× @ 10K**; **tie @ 100K–1M**  
- GPU vs OpenMP kernel @ 1M: **~10×**  
- Wall @ 1M: **~391 ms** (transfer dominates) ≈ FAISS CPU

### 7.2 dim=1024 (BGE-large width), kernel-only

| N | CPU OMP | GPU naive | GPU tiled | FAISS |
|---|---|---|---|---|
| 10K | 8.11 ms | 0.60 ms | **0.38 ms** | 11.2 ms |
| 100K | 86.4 ms | 6.40 ms | 7.27 ms | 123 ms |
| 1M | 893 ms | 65.2 ms | 66.6 ms | OOM |

Same pattern: query staging wins at small N; candidate dominates at large N.

### 7.3 PyTorch extension (device-resident, `torch.cuda.Event`)

| N | custom L2 | torch.cdist | custom cosine | F.cosine_sim |
|---|---|---|---|---|
| 10K | 0.24 ms | 6.15 ms | 0.24 ms | 3.32 ms |
| 100K | 2.47 ms | 4.67 ms | 2.20 ms | 5.13 ms |
| 1M | 20.8 ms | 48.8 ms | 20.9 ms | 50.9 ms |

**~1.9× vs cdist @ 100K, ~2.4× @ 1M** — narrow shape, T4, float32 only.

---

## 8. PyTorch → CUDA call chain (extension path)

```
Python  vst.l2_distance(query, store)
  → vector_search_torch.py  (.contiguous(), flatten query)
  → import vector_search_torch_cpp  (pybind11 module)
  → binding.cpp  tiled_l2_distance()
       check_inputs: CUDA, float32, shape, contiguous
       out = torch::empty({N}, query.options())
       data_ptr<float>() on store, query, out
  → tiled_kernels.cu  launch_l2_tiled()
       l2_tiled_kernel<<<blocks, 256, dim*4>>>(...)
  → return Tensor (async on current CUDA stream)
```

**vs bench_runner:** `cudaMalloc` + H2D every call — intentional to expose transfer cost.

**Benchmark sync:** `torch.cuda.synchronize()` in `benchmark_torch.py`; binding has **no** `cudaDeviceSynchronize` in inference path.

---

## 9. L2 vs cosine vs RAG metric

| | **L2** (this bench primary) | **Cosine** | **RAG / FAISS path** |
|---|---|---|---|
| Formula | `sqrt(Σ(q-c)²)` | `dot / (‖q‖‖c‖)` | FAISS FlatL2 on embeddings |
| If vectors L2-normalized | Related to cosine | Common for semantic search | BGE outputs often normalized |
| In repo | Both implemented | Both implemented | RAG uses L2 index on CPU |

**No ML regularization** in this repo — L2 = distance metric; `fillRandom(normalize=true)` is **unit-vector test data**, not weight decay.

---

## 10. Verification & timing workflow

### Before any benchmark number

```bash
./test_correctness_gpu      # GPU vs CPU, tol 1e-4
./test_correctness          # CPU sanity
cd pytorch_ext && python test_correctness_torch.py
```

### What we check

- GPU naive & tiled vs CPU single-thread  
- Naive vs tiled cross-check  
- Edge cases: **n=257** (non-block-aligned)  
- PyTorch: vs `torch.cdist` / `F.cosine_similarity` (tol 1e-3)

### Timing layers

| Layer | Tool | What it includes |
|---|---|---|
| Kernel-only | `cudaEvent` / `torch.cuda.Event` | GPU compute only |
| Wall | `std::chrono` / end-to-end Python | malloc, H2D, kernel, D2H |
| Not done | — | warm-up, median, p95, Nsight |

---

## 11. STAR stories (ready to tell)

### Story A — Shared memory win that doesn’t scale

- **S:** Brute-force L2 for RAG-style retrieval on GPU.  
- **T:** Cut redundant memory traffic without losing correctness.  
- **A:** Profiled access mentally (Roofline); staged query in shared memory; A/B naive vs tiled; separated kernel vs wall time.  
- **R:** **1.6× @ 10K**; **no win @ 1M** — proved candidate reads dominate; honest README.

### Story B — PyTorch extension for realistic serving

- **S:** `bench_runner` re-uploads corpus every query — unrealistic.  
- **T:** Measure kernel like inference (device-resident).  
- **A:** pybind11 extension; validate tensors; launch on PyTorch stream.  
- **R:** Device-resident custom L2 **beats cdist ~2× @ 1M** on T4 for this narrow case.

### Story C — Correctness before performance

- **S:** First GPU port of distance kernel.  
- **T:** Trust nothing until verified.  
- **A:** CPU ground truth; fp tolerance 1e-4; boundary n=257; naive vs tiled agreement.  
- **R:** Caught that “fast but wrong” is useless for retrieval.

---

## 12. Mock Q&A (short answers)

**Q: Why not just use cuBLAS / torch.cdist?**  
A: Learning exercise + controlled A/B. Production PyTorch ops are highly tuned; we compare honestly on device-resident brute-force L2 for one shape.

**Q: Why did tiled stop helping at 1M?**  
A: Optimization removed redundant **query** reads; at large N each candidate still reads **D unique floats** — that dominates.

**Q: Memory-bound or compute-bound?**  
A: Roofline memory-bound (AI ≈ 0.75). But not DRAM-saturated — ~15–20% effective useful BW due to strided access.

**Q: What’s next?**  
A: Block-per-vector reduction for coalescing along dim; candidate tiling; batch queries; device-resident index (PyTorch path started this).

**Q: How does this relate to your RAG project?**  
A: RAG uses FAISS CPU FlatL2 in production-shaped demo; CUDA repo explores GPU primitive underneath — companion, not replacement.

**Q: Top-k on GPU?**  
A: Thrust **full sort_by_key** — engineering trade-off, not optimized partial top-k.

**Q: Determinism / FP associativity?**  
A: Serial fp32 accumulation today. Parallel tree reduction would change sum order — same issue as PyTorch deterministic mode.

---

## 13. “Walk me through the kernel” (60 sec script)

> One block has 256 threads; each thread owns one candidate index `i`. It loads the query — in the naive kernel, every thread re-reads the full query from global memory. In the tiled kernel, the block cooperatively loads the query into shared memory once, syncs, then each thread streams its candidate row from global memory and accumulates squared differences in a serial loop, then sqrt. Launch is `<<<ceil(N/256), 256, dim*sizeof(float)>>>`. I verify against CPU with 1e-4 tolerance before benchmarking. Roofline says memory-bound; at 10K shared memory helps; at 1M candidate traffic dominates so naive and tiled tie.

---

## 14. Reduction lecture paragraph (reuse verbatim if asked)

> Reductions collapse a vector to a scalar via identity + binary op — why PyTorch uses generic reduction kernels. Parallel form is a tree halving active elements each step. Optimization order: fix **warp divergence** (contiguous active lanes) → **shared memory** to cut global traffic → **multiblock + atomics** → **thread coarsening**. Subtlety: **fp non-associativity** — sum order matters; deterministic mode costs performance; fp16 often accumulates in fp32.

**Map to project:** inner L2 loop is a reduction over `d`; we parallelize over **candidates**, not **dimensions** — that’s the coalescing trade-off.

---

## 15. Don’t-say / do-say card

| ❌ Don’t | ✅ Do |
|---|---|
| Beat FAISS in general | Beat **FAISS CPU IndexFlatL2** on kernel-only brute-force |
| Saturated DRAM | Memory-bound but **~15–20%** effective useful BW |
| Full reduction pipeline implemented | Shared-query done; block-reduction **planned** |
| Shared tiling = GEMM | **Query-only** staging, not 2D candidate tile |
| Thrust partial top-k | Thrust **full sort** |
| Nsight profiling | `cudaEvent` + Roofline reasoning |
| Regularization in project | L2 **metric**; L2-normalize **test data** only |

---

## 16. Run commands (Colab / interview demo)

```bash
# Build (Colab: cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..)
mkdir build && cd build && cmake .. && make -j
./test_correctness_gpu && ./bench_runner

# FAISS baseline
pip install faiss-cpu numpy
python3 scripts/faiss_baseline.py --dim 384 --queries 50

# PyTorch extension
cd pytorch_ext && pip install -e .
python test_correctness_torch.py && python benchmark_torch.py
python demo_rerank.py
```

---

## 17. Financial RAG quick facts (companion project)

Use when interview shifts to **system design**:

- **Stack:** Streamlit UI, Mistral LLM, **BGE-large** embeddings (1024-d), **FAISS CPU IndexFlatL2**, Docker.  
- **Retrieval:** exact brute-force L2 on CPU — same *class* of search as this CUDA repo.  
- **Why CPU FAISS in RAG:** pragmatic for demo scale; IVF/HNSW/PQ for production scale at millions+.  
- **CUDA repo role:** “I wanted to understand the GPU primitive my RAG stack relies on at the library level.”

---

## 18. One-page mental model (draw on whiteboard)

```
[Query q] ──┐
            ├──► for each candidate c in store[N×D]:
[Store]  ───┘         dist = L2(q, c)   ← THIS REPO optimizes this loop
                           │
                           ▼
                      top-k indices  ← CPU partial_sort / Thrust sort
                           │
                           ▼
                      LLM context    ← Financial RAG
```

**Bottleneck today:** strided global read of `store[i*D : i*D+D]` per thread.  
**Fix shipped:** cache `q` in shared memory per block.  
**Fix next:** cooperate along `D` (block-per-vector reduction).

---

## 19. Checklist night-before

- [ ] Say **3D+1 FLOPs**, AI **0.75**, ridge **25–27**  
- [ ] **1.6× @ 10K**, **tie @ 1M**, **~10× vs OMP @ 1M kernel**  
- [ ] Wall **~391 ms** vs kernel **~28 ms** @ 1M  
- [ ] Run **`test_correctness_gpu` first**  
- [ ] Shared-query ≠ parallel reduction ≠ GEMM  
- [ ] PyTorch path: **pybind11 → data_ptr → <<<>>>**  
- [ ] No regularization; FAISS = **FlatL2 CPU** baseline only  
- [ ] Next: **block-per-vector** for coalescing  

---

*Last aligned with repo README + Colab T4 results in `results/benchmark*.csv`.*
