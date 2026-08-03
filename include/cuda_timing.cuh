#pragma once

// Shared timing state so both the naive and tiled kernel implementations
// can report "time of the last kernel launch" through the same
// last_kernel_time_ms() function declared in distance_cuda.cuh, without
// each .cu file defining its own copy of that symbol (which would be an
// ODR violation at link time).
void set_last_kernel_time_ms(float ms);
