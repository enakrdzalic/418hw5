// Functions which must be defined by moments-kernel.cu.

// Capped at 7 to avoid running out of shared memory when using
// maximum block size of 1024.
#define MAX_MOMENT 7

// Must return a value between 10 and 5e8.
uint best_n();

// Must return a value between 2 and MAX_MOMENT.
uint best_moment();

// Solution should work for 10 <= n <= 5e8 and 2 <= max_moment <= MAX_MOMENT.
//
// Note that the d_x and d_moments arguments are device pointers --
// the input data has already been loaded onto the GPU, and output
// should remain there at the end of the function call.
void compute_all_gpu(const float *d_x, const uint n,
		     const uint max_moment, double *d_moments);
