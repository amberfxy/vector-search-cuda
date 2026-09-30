#pragma once

// Optional NVTX ranges for Nsight Systems / Compute timelines.
// When VECTOR_SEARCH_USE_NVTX is not defined, all macros are no-ops so
// CPU-only and default CUDA builds stay unchanged.

#if defined(VECTOR_SEARCH_USE_NVTX)
  #if defined(__has_include)
    #if __has_include(<nvtx3/nvToolsExt.h>)
      #include <nvtx3/nvToolsExt.h>
    #elif __has_include(<nvToolsExt.h>)
      #include <nvToolsExt.h>
    #else
      #undef VECTOR_SEARCH_USE_NVTX
    #endif
  #else
    #include <nvToolsExt.h>
  #endif
#endif

#if defined(VECTOR_SEARCH_USE_NVTX)
  #define VS_NVTX_RANGE_PUSH(name) nvtxRangePushA(name)
  #define VS_NVTX_RANGE_POP()      nvtxRangePop()
#else
  #define VS_NVTX_RANGE_PUSH(name) ((void)0)
  #define VS_NVTX_RANGE_POP()      ((void)0)
#endif

// RAII helper for C++ scopes.
struct NvtxRange {
    explicit NvtxRange(const char* name) { VS_NVTX_RANGE_PUSH(name); }
    ~NvtxRange() { VS_NVTX_RANGE_POP(); }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};
