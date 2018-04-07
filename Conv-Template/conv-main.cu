#include <stdio.h>
#include <string.h>
#include <math.h>
//#include <assert.h>
#include "cpu-helper.h"
#include "cuda-helper.h"
#include "timing418.h"
#include "conv-helper.h"
#include "conv-kernel.h"

// Difference tolerated between two floating point numbers that are
// considered "equal".  We chose this value as slightly smaller than
// 1/256 and we are writing to an output image format with 256 grey levels.
#define CLOSE_ENOUGH 1e-3

// Print the input, mask and output arrays?
#define DEBUG_PRINT 0

// Save the input, mask and output arrays to PGM files?
#define DEBUG_SAVE 1

// Compare CPU and basic GPU solutions?  (Otherwise compare basic and
// tiled GPU versions).
#define COMPARE_WITH_CPU 1

//----------------------------------------------------------------------
// Confirm that the entries of two arrays are the same (to within
// CLOSE_ENOUGH) and that neither contains NaN values.
void check_float_array(const float *x1, const float *x2,
		       const uint m, const uint n,
		       const char *s1, const char *s2) {

  for(uint i = 0; i < m; i++)
    for(uint j = 0; j < n; j++) {
    if(fabs(x1[IDX2F(i,j,m,n)] - x2[IDX2F(i,j,m,n)]) > CLOSE_ENOUGH) {
      fprintf(stderr, "\nERROR: element=(%d,%d) %s=%f, %s=%f\n",
              i, j, s1, x1[IDX2F(i,j,m,n)], s2, x2[IDX2F(i,j,m,n)]);
      exit(-1);
    }
    if(isnan(x1[IDX2F(i,j,m,n)]) || isnan(x2[IDX2F(i,j,m,n)])) {
      fprintf(stderr, "\nNaN detected: element=(%d,%d) %s=%f, %s=%f\n",
              i, j, s1, x1[IDX2F(i,j,m,n)], s2, x2[IDX2F(i,j,m,n)]);
      exit(-1);
    }
  }
}

//----------------------------------------------------------------------
// Print an array to the screen.  Only really useful for small arrays.
void print_array(const float *x, const uint m, const uint n, const char *s) {

  printf("Array %s (%d x %d): ", s, m, n);
  for(uint i = 0; i < m; i++) {
    printf("\n  ");
    for(uint j = 0; j < n; j++)
      printf("%4.2e ", x[IDX2F(i,j,m,n)]);
  }
  printf("\n");
}

//----------------------------------------------------------------------
// Computes the convolution of data with mask, and stores the
// result to data_out.
//
// Assumes zero boundary condition: Data outside the input data set is
// assumed to be zero.
//
// Precondition: p and q (mask size) are both odd.
void conv_cpu(const float *data, float *data_out,
	      const uint m, const uint n,
	      const float *mask, const uint p, const uint q) {

  // Integer divide causes round down assuming precondition is met.
  const uint p_half = p / 2;
  const uint q_half = q / 2;
  
  // Loop over all input elements.  Because we are storing data
  // column-major, we want to iterate over rows on the inner-most
  // loops to get the best benefit from spatial locality.
  for(uint j = 0; j < n; j++) 
    for(uint i = 0; i < m; i++) {
      float sum = 0.0;
      for(uint l = 0; l < q; l++) {
      	// Which column of input data are we examining?
      	const int j_offset = j - q_half + l;
      	if((j_offset >= 0) && (j_offset < n))
      	  // If this is a valid column.
      	  for(uint k = 0; k < p; k++) {
      	    // Which row of input data are we examining?
      	    const int i_offset = i - p_half + k;
      	    if((i_offset >= 0) && (i_offset < m))
      	      // If this is a valid row.
      	      sum += (data[IDX2F(i_offset,j_offset,m,n)]
      		      * mask[IDX2F(k,l,p,q)]);
      	  }
      }
      // Save the result.
      data_out[IDX2F(i,j,m,n)] = sum;
    }
}

    
//----------------------------------------------------------------------
/* Compute the FP operations for computing.  Note that this
   calculation is a little unfair, since it counts division the same
   as all the other operations, even though it has latency about 50x
   that of addition or multiplication; however, it should be noted
   that division latency is comparable to global memory latency and we
   need to write the final result back to global memory whether we
   divide or not.

   Assume that m*n >> p*q, so boundary condition can be ignored.
   Assume p*q >> 1, so final division can be ignored.
*/

uint total_fp_ops(const uint m, const uint n, const uint p, const uint q) {

  // Each output element (m*n elements) requires convolution with mask
  // (p*q elements).
  return (2 * m * n * p * q);
}

//----------------------------------------------------------------------
// Run the convolution on both the CPU and GPU so that we can compare
// the results).  Compute timings.  Possibly print the results to the
// screen and/or save to a PGM file.
void conv_protocol(const float *data, const uint m, const uint n, 
		   const uint p, const uint q) {

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  // Always use odd number of trials so that median is easy to calculate.
  const uint trials = 1;
  timer_418 t0, t1;

  // Place to store the data.
  float *mask, *out1, *out2;
  double *run_times;

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  // Allocate the vectors.
  NewArrayCPU(float, p*q, mask);
  NewArrayCPU(float, m*n, out1);
  NewArrayCPU(float, m*n, out2);
  NewArrayCPU(double, trials, run_times);

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  // Fill the mask.  

  // Constant mask for maximum (rectangular) blurring.  We divide
  // through by the size of the mask so that the entries sum to 1.0
  // and we get roughly the same input and output scaling.
  const_float_array(mask, p, q, p, 1.0 / (p*q));

  // Horizontal blurring mask.
  //one_spot_array(mask, p, q, p, p/2, 0, p/2+1, q, 0.0, 1.0 / q);

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  // Print the data (for debugging)
  if(DEBUG_PRINT) {
    printf("\n");
    print_array(data, m, n, "input data");
    printf("\n");
    print_array(mask, p, q, "mask");
  }

  if(DEBUG_SAVE) {
    write_pgm(data, m, n, "save-data.pgm");
    write_pgm(mask, p, q, "save-mask.pgm");
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  // Compute the convolution on the CPU.
  if(COMPARE_WITH_CPU) {
    printf("\nConvolution on CPU: ");
    for(uint i = 0; i < trials; i++) {
      timestamp(&t0);
      conv_cpu(data, out1, m, n, mask, p, q);
      timestamp(&t1);
      run_times[i] = elapsed_seconds(t1, t0);
      printf(" %5.2e", run_times[i]);
    }
    double median_time_cpu = median_time(run_times, trials);
    printf("\n  -- median %5.2e, flops %5.2e",
	   median_time_cpu, total_fp_ops(m, n, p, q) / median_time_cpu);
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  // Copy data over to the GPU.
  float *d_data, *d_out1, *d_out2;
  NewArrayGPU(float, m*n, d_data);
  NewArrayGPU(float, m*n, d_out1);
  NewArrayGPU(float, m*n, d_out2);
  cudaTry(cudaMemcpy(d_data, data, m*n*sizeof(float),cudaMemcpyHostToDevice)); 
  set_up_mask(mask, p, q);

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  printf("\n\nConvolution on GPU (basic): ");
  for(uint i = 0; i < trials; i++) {
    timestamp(&t0);
    // Launch the kernel to compute convolution on the GPU.
    conv_gpu(d_data, d_out2, m, n, false);
    // Make sure the kernel has completed.
    cudaDeviceSynchronize();
    timestamp(&t1);
    run_times[i] = elapsed_seconds(t1, t0);
    printf(" %5.2e", run_times[i]);
  }
  double median_time_gpu_basic = median_time(run_times, trials);
  printf("\n  -- median %5.2e, flops %5.2e",
	 median_time_gpu_basic, 
	 total_fp_ops(m, n, p, q) / median_time_gpu_basic);

  // Retrieve data from the GPU.
  cudaTry(cudaMemcpy(out2, d_out2, m*n*sizeof(float), cudaMemcpyDeviceToHost));

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  if(!COMPARE_WITH_CPU) {
    printf("\n\nConvolution on GPU (tiled): ");
    for(uint i = 0; i < trials; i++) {
      timestamp(&t0);
      // Launch the kernel to compute convolution on the GPU.
      conv_gpu(d_data, d_out1, m, n, true);
      // Make sure the kernel has completed.
      cudaDeviceSynchronize();
      timestamp(&t1);
      run_times[i] = elapsed_seconds(t1, t0);
      printf(" %5.2e", run_times[i]);
    }
    double median_time_gpu_tiled = median_time(run_times, trials);
    printf("\n  -- median %5.2e, flops %5.2e",
	   median_time_gpu_tiled, 
	   total_fp_ops(m, n, p, q) / median_time_gpu_tiled);
    // Retrieve data from the GPU.
    cudaTry(cudaMemcpy(out1, d_out1, m*n*sizeof(float), 
		       cudaMemcpyDeviceToHost));
  }

  // Check the two different output 
  if(COMPARE_WITH_CPU)
    check_float_array(out1, out2, m, n, "CPU", "GPU (basic)");
  else
    check_float_array(out1, out2, m, n, "GPU (tiled)", "GPU (basic)");

  // Free up the arrays on the GPU.
  cudaTry(cudaFree(d_data));
  cudaTry(cudaFree(d_out1));
  cudaTry(cudaFree(d_out2));
  clean_up_mask();
  
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  // Show the results.
  if(DEBUG_PRINT) {
    printf("\n");
    print_array(out2, m, n, "data out");
  }

  if(DEBUG_SAVE)
    write_pgm(out2, m, n, "save-out.pgm");

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  // Free up all the CPU memory resources.
  free(mask);
  free(out1);
  free(out2);
  free(run_times);
}


//----------------------------------------------------------------------
// Parses the input arguments and loads the input data set.
int main(int argc, char **argv) {

  const uint p_default = 21, q_default = 21;
  uint m, n, p, q;
  char *conversion_garbage;
  bool usage_two = false;
  float *data;

  if((argc < 2) || (argc > 5)) {
    fprintf(stderr, "\nusage: %s m [ n = m ] [ p = %d ] [ q = %d ]",
	    argv[0], p_default, q_default);
    fprintf(stderr, "\n   or: %s filename.pgm\n\n", argv[0]);
    exit(1);
  }

  // Parse the input arguments.

  // Start by assuming the first usage option.
  m = strtoul(argv[1], &conversion_garbage, 10);
  if(strlen(conversion_garbage) > 0) {
    // If some portion of the first argument could not be converted to
    // an integer, we may want to switch to the second usage option.
    if(argc != 2) {
      // Second usage option only allows a single argument.
      fprintf(stderr, "%s unable to convert %s to uint m\n", argv[0], argv[1]);
      exit(1);
    }
    else {
      // Second usage option: Load the file.  This call will set m and
      // n parameters, and parameters p and q will be set to their
      // defaults because argc == 2.
      read_pgm(&data, &m, &n, argv[1]);
      usage_two = true;
    }
  }

  if(argc > 2) {
    n = strtoul(argv[2], &conversion_garbage, 10);
    if(strlen(conversion_garbage) > 0) {
      fprintf(stderr, "%s unable to convert %s to uint m\n", argv[0], argv[2]);
      exit(1);
    }
  }
  else
    n = m;
  
  if(argc > 3) {
    p = strtoul(argv[3], &conversion_garbage, 10);
    if(strlen(conversion_garbage) > 0) {
      fprintf(stderr, "%s unable to convert %s to uint p\n", argv[0], argv[3]);
      exit(1);
    }
  }
  else
    p = p_default;

    
  if(argc > 4) {
    q = strtoul(argv[4], &conversion_garbage, 10);
    if(strlen(conversion_garbage) > 0) {
      fprintf(stderr, "%s unable to convert %s to uint q\n", argv[0], argv[4]);
      exit(1);
    }
  }
  else
    q = q_default;

  // Uncomment the call to srand() if you want different random number
  // sequences on each run.
  // srand((uint)time(NULL));

  // Allocate space for and fill the input data (if it wasn't read
  // from an image).
  if(!usage_two) {
    NewArrayCPU(float, m*n, data);

    // Uncomment one of the cases below to create an initial condition.

    // A single box in the middle of the data.
    one_spot_array(data, m, n, m, m/3, n/3, 2*m/3, 2*n/3, 0.0f, 1.0f);

    // Vertical stripes.
    //stripe_vert_array(data, m, n, m, n/10, n/5, 0.0f, 1.0f);

    // Horizontal stripes.
    //stripe_hori_array(data, m, n, m, m/6, m/12, -1.5f, +0.5f);

    // Checkerboard.
    //checkerboard_array(data, m, n, m, m/8, m/8, n/8, n/8, 0.0f, 1.0f);
  }

  printf("Computing convolution of %d x %d input data with %d x %d mask\n",
	 m, n, p, q);
  conv_protocol(data, m, n, p, q);
  printf("\n\n");

  free(data);
}
