#include <stdio.h>
#include <stdlib.h>
//#include <cuda_runtime.h>
#include "cuda-helper.h"
#include "cpu-helper.h"
#include "timing418.h"
#include "conv-kernel.h"

#define BLOCK_SIZE 512
#define MAX_MASK_WIDTH 41
#define MAX_MASK_HEIGHT 41
#define TILE_WIDTH 32

__constant__ float M[MAX_MASK_WIDTH*MAX_MASK_HEIGHT];
__constant__ uint M_WIDTH;
__constant__ uint M_HEIGHT;

//----------------------------------------------------------------------
// We will use these values when confirming the optimal throughput of
// your tiled kernel.

// Must return a value between 500 and 15000.
uint best_mn() {
  return 1024;
}

// Must return a value between 5 and 21.
uint best_pq() {
  return 21;
}

//----------------------------------------------------------------------
void set_up_mask(const float *mask, const uint p, const uint q) {	
	cudaTry(cudaMemcpyToSymbol(M_HEIGHT, &p, sizeof(uint)));
	cudaTry(cudaMemcpyToSymbol(M_WIDTH, &q, sizeof(uint)));
	cudaTry(cudaMemcpyToSymbol(M, mask, p*q*sizeof(float)));
}

//----------------------------------------------------------------------
void clean_up_mask() {

}

__global__ void conv_tiled_kernel(const float *d_data_in, 
		  float *d_data_out, const uint m, const uint n) {
	// The in & out arrays are m height by n width
	
	// Calculate x and y indices of output element
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y*TILE_WIDTH + ty;
	int col_o = blockIdx.x*TILE_WIDTH + tx;

	int row_i, col_i;

	__shared__ float N_ds[TILE_WIDTH + MAX_MASK_WIDTH - 1]
				[TILE_WIDTH + MAX_MASK_HEIGHT - 1];

	// Load elements to the top left
	row_i = row_o - (M_HEIGHT/2);
	col_i = col_o - (M_WIDTH/2);

	// Only top M_HEIGHT/2 threads should load top M_HEIGHT/2 rows
	if (ty < M_HEIGHT/2) {
		if ((row_i >= 0) && (row_i < m) && (col_i >= 0) && (col_i < n)) {
			N_ds[ty][tx] = d_data_in[row_i*n + col_i];
		} else {
			N_ds[ty][tx] = 0;
		}
	}

	// Load elements to the top right
	row_i = row_o - (M_HEIGHT/2);
	col_i = col_o + (M_WIDTH/2);

	// Only top M_HEIGHT/2 threads should load top M_HEIGHT/2 rows
	if (ty < M_HEIGHT/2) {
		if ((row_i >= 0) && (row_i < m) && (col_i >= 0) && (col_i < n)) {
			N_ds[ty][tx] = d_data_in[row_i*n + col_i];
		} else {
			N_ds[ty][tx] = 0;
		}
	}

	// Load elements to the left
	row_i = row_o;
	col_i = col_o - (M_WIDTH/2);

	if (tx < M_WIDTH/2) {
		if ((row_i >= 0) && (row_i < m) && (col_i >= 0) && (col_i < n)) {
			N_ds[ty][tx] = d_data_in[row_i*n + col_i];
		} else {
			N_ds[ty][tx] = 0;
		}
	}

	// Load inner elements
	if (tx < n && ty < m) {
		N_ds[ty][tx] = d_data_in[row_o*n + col_o];
	}

	// Load elements to the right
	row_i = row_o;
	col_i = col_o + (M_WIDTH/2);

	if (tx < M_WIDTH/2) {
		if ((row_i >= 0) && (row_i < m) && (col_i >= 0) && (col_i < n)) {
			N_ds[ty][tx] = d_data_in[row_i*n + col_i];
		} else {
			N_ds[ty][tx] = 0;
		}
	}

	// Load elements to the bottom left
	row_i = row_o + (M_HEIGHT/2);
	col_i = col_o - (M_WIDTH/2);

	// Only top M_HEIGHT/2 threads should load bottom M_HEIGHT/2 rows
	if (ty < M_HEIGHT/2) {
		if ((row_i >= 0) && (row_i < m) && (col_i >= 0) && (col_i < n)) {
			N_ds[ty][tx] = d_data_in[row_i*n + col_i];
		} else {
			N_ds[ty][tx] = 0;
		}
	}

	// Load elements to the bottom right
	row_i = row_o + (M_HEIGHT/2);
	col_i = col_o + (M_WIDTH/2);

	// Only top M_HEIGHT/2 threads should load top M_HEIGHT/2 rows
	if (ty < M_HEIGHT/2) {
		if ((row_i >= 0) && (row_i < m) && (col_i >= 0) && (col_i < n)) {
			N_ds[ty][tx] = d_data_in[row_i*n + col_i];
		} else {
			N_ds[ty][tx] = 0;
		}
	}

	__syncthreads();

	// Calculate output 

	row_i = row_o - (M_HEIGHT/2);
	col_i = col_o - (M_WIDTH/2);

	float output = 0;
	
	for (int i=0; i<M_HEIGHT; i++) {
		for (int j=0; j<M_WIDTH; j++) {
			output += M[i*M_WIDTH + j] * N_ds[row_i+i][col_i+j];
		}
	}

	d_data_out[row_o*n + col_o] = output;
}

void conv_tiled_gpu(const float *d_data_in, float *d_data_out,
	      const uint m, const uint n, const bool tiled) {
	// Each thread should load 1 element from d_data_in into shared mem
	// The in & out arrays are m height by n width

	// TODO: Is this right?
	dim3 dimGrid(ceil((float)(n/TILE_WIDTH)), ceil((float)(m/TILE_WIDTH)), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    conv_tiled_kernel<<<dimGrid, dimBlock>>>(d_data_in, d_data_out, m, n);
}

__global__ void conv_basic_kernel(const float *d_data_in, 
		  float *d_data_out, const uint m, const uint n) {
	// The in & out arrays are m height by n width

	int i = blockDim.x*blockIdx.x + threadIdx.x;

	if (i < m*n) {

		int x_start = (i % n) - (M_WIDTH/2);
		int y_start = (i / n) - (M_HEIGHT/2);

		float out_value = 0;

		for (int j=0; j<M_HEIGHT; j++) {
			for (int k=0; k<M_WIDTH; k++) {

				if ((y_start + j) >= 0 && (y_start + j) < m &&
						(x_start + k) >= 0 && (x_start + k) < n) {

					int data_index = ((y_start + j) * n) + (x_start + k);

					if (data_index >= 0 && data_index < n*m) {

						int mask_index = (j * M_WIDTH) + k;
						out_value += d_data_in[data_index] * M[mask_index];
					}
				}
			}
		}

		d_data_out[i] = out_value;
	}
}

void conv_basic_gpu(const float *d_data_in, float *d_data_out,
	      const uint m, const uint n, const bool tiled) {
	// Each thread should calculate 1 output element
	// The in & out arrays are m height by n width
	// Thus we should create m*n threads
	// Since 1 block has 1024 threads, we should create n*m/1024 blocks

	dim3 dimGrid(ceil((float)(n*m)/(float)BLOCK_SIZE), 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    conv_basic_kernel<<<dimGrid, dimBlock>>>(d_data_in, d_data_out, m, n);
}

//----------------------------------------------------------------------
// Computes the convolution on the GPU in the very basic manner: One
// thread per output pixel with all data read or written directly to
// global memory.
void conv_gpu(const float *d_data_in, float *d_data_out,
	      const uint m, const uint n, const bool tiled) {
	if (tiled) 
		conv_tiled_gpu(d_data_in, d_data_out, m, n, tiled);
	else 
		conv_basic_gpu(d_data_in, d_data_out, m, n, tiled);
}
