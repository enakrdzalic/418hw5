// Functions which must be defined by conv-kernel.cu

// Must return a value between 500 and 15000.
uint best_mn();

// Must return a value between 5 and 21
uint best_pq();

// Call this before attempting any convolution.
void set_up_mask(const float *mask, const uint p, const uint q);

// Call this when you are done with the convolution.
void clean_up_mask();

// Call this to execute the convolution on the GPU.  Note that the
// input and output data are already copied to the GPU (d_data_in and
// d_data_out are GPU arrays).  Also, there is no need to pass the
// mask because it was already set by calling set_up_mask().
void conv_gpu(const float *d_data_in, float *d_data_out,
	      const uint m, const uint n, const bool tiled);
