#include "cuda-helper.h"

//----------------------------------------------------------------------
// Allocate memory on the GPU and copy over the input data.
void data_to_gpu(const float *x, const uint n, double max_moment,
		 float **d_x, double **d_moments) {


  // Allocate memory on the GPU for all of the arrays.
  NewArrayGPU(float, n, *d_x);
  NewArrayGPU(double, max_moment, *d_moments);

  // Copy over the input arrays.
  cudaTry(cudaMemcpy(*d_x, x, n * sizeof(float), cudaMemcpyHostToDevice)); 
}

//----------------------------------------------------------------------
// In case there is any work that needs to be done after calling
// compute_all_gpu().
void after_run_gpu() {

  // Now we need to explicitly wait for the kernels to complete,
  // otherwise the CPU will not wait for the second kernel to finish.
  cudaDeviceSynchronize();
}

//----------------------------------------------------------------------
// Copy over the output data and free memory on the GPU.
void data_from_gpu(double *moments, double max_moment,
		   float *d_x, double *d_moments) {

  // Copy back the result.
  cudaTry(cudaMemcpy(moments, d_moments, max_moment * sizeof(double),
		     cudaMemcpyDeviceToHost));

  // Free up the arrays on the GPU.
  cudaTry(cudaFree(d_x));
  cudaTry(cudaFree(d_moments));
}
