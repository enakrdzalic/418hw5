#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "moments-kernel.h"

#define BLOCK_SIZE 1024

// Must return a value between 10 and 5e8.
uint best_n() {
	return 40000000;
}

// Must return a value between 2 and MAX_MOMENT.
uint best_moment() {
	return 7;
}

__global__ void compute_mean_kernel(const float *d_x, const uint n, double *d_temp) {

  	int i = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int t = threadIdx.x;

  	if (i < n) {
		__shared__ double partialCompute[BLOCK_SIZE];
		partialCompute[t] = (double) d_x[i];
	  	
	  	for (unsigned int stride = blockDim.x/2; stride >= 1; stride = stride >> 1) {
		  	__syncthreads();
	  		if (t < stride) {
	  			partialCompute[t] += partialCompute[t+stride];
	  		}
	  	}

	  	__syncthreads();

	  	// One thread moves it back
	  	if (t < 1) {
		  	d_temp[blockIdx.x] = partialCompute[0];
	  	}
  	}
}

__global__ void compute_moments_kernel(const float *d_x, const uint n, 
			 int moment, double *d_moments, double *d_temp) {

  	int i = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int t = threadIdx.x;

  	if (i < n) {
		__shared__ double partialCompute[BLOCK_SIZE];
		partialCompute[t] = pow((double)(d_x[i] - d_moments[0]), (double)moment);
	  	
	  	for (unsigned int stride = blockDim.x/2; stride >= 1; stride = stride >> 1) {
		  	__syncthreads();
	  		if (t < stride) {
	  			partialCompute[t] += partialCompute[t+stride];
	  		}
	  	}

	  	__syncthreads();

	  	// One thread moves it back
	  	if (t < 1) {
		  	d_temp[blockIdx.x] = partialCompute[0];
	  	}
  	}
}

__global__ void divide_n_kernel(const uint n, double *d_moments, 
			 double *d_temp, double m, int index) {

	unsigned int t = threadIdx.x;

	__shared__ double partialCompute[BLOCK_SIZE];
	partialCompute[t] = 0;

	double elements_per_thread = ceil((double)m/(double)blockDim.x);

	int start = t*elements_per_thread;
	int stop = (t+1)*elements_per_thread;

	stop = (stop < m) ? stop : m;

	if (start < m) {
		for (int i=start; i<stop; i++) {
			partialCompute[t] += d_temp[i];
		}
	}

	__syncthreads();

	if (t < 1) {
		double sum = 0;
		for (int i=0; i<blockDim.x; i++) {
			sum += partialCompute[i];
		}
		sum = sum/n;
		d_moments[index] = sum;
	}
}

// Solution should work for 10 <= n <= 5e8 and 2 <= max_moment <= MAX_MOMENT.
//
// Note that the d_x and d_moments arguments are device pointers --
// the input data has already been loaded onto the GPU, and output
// should remain there at the end of the function call.
void compute_all_gpu(const float *d_x, const uint n,
		     const uint max_moment, double *d_moments) {
    double *d_temp;

	double numBlocks = ceil((float)n/(float)BLOCK_SIZE);
	int size = numBlocks*sizeof(double);

    cudaMalloc((void**)(&d_temp), size);

	dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    // Sum up all elements of d_x & put in d_temp
    compute_mean_kernel<<<dimGrid, dimBlock>>>(d_x, n, d_temp);
    divide_n_kernel<<<1, dimBlock>>>(n, d_moments, d_temp, numBlocks, 0);

	for (int moment = 2; moment <= max_moment; moment++) {
    	compute_moments_kernel<<<dimGrid, dimBlock>>>(d_x, n, moment, d_moments, d_temp);
    	divide_n_kernel<<<1, dimBlock>>>(n, d_moments, d_temp, numBlocks, moment-1);
    }
}