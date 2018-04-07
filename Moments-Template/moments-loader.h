// Functions to allocate buffers on the GPU, load the input data,
// retrieve the output data and free the buffers.

void data_to_gpu(const float *x, const uint n, double max_moment,
		 float **d_x, double **d_moments);

void after_run_gpu();

void data_from_gpu(double *moments, double max_moment,
		   float *d_x, double *d_moments);
