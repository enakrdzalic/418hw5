#include <stdio.h>
#include <math.h>
#include "cpu-helper.h"
#include "moments-loader.h"
#include "moments-kernel.h"
#include "timing418.h"

//--------------------------------------------------------------------------
// Random floats between lower and upper (inclusive).  Note that we
// construct the number using double precision.
void rand_float_vector(float *x, const uint n, 
		       const double lower, const double upper) {

  double gap = upper - lower;
  
  for(uint i = 0; i < n; i++) {
    double unit_uniform = (double)rand() / (double)RAND_MAX;
    x[i] = (float)(gap * unit_uniform + lower);
  }
}

void compute_all_cpu(const float *x, const uint n,
         const uint max_moment, double *moments) {
  // First compute E[X]
  for (int i=0; i<(int)n; i++) {
    moments[0] += x[i];
  }
  moments[0] = moments[0]/n;

  // Then compute all the individual moments
  for (int moment=2; moment<=(int)max_moment; moment++) {
    for (int i=0; i<(int)n; i++) {
      moments[moment-1] += pow((double)(x[i]-moments[0]), moment);
    }
    moments[moment-1] = moments[moment-1]/(double)n;
  }
}

uint total_fp_ops(const uint n, const uint max_moment) {

  // Each output element (m*n elements) requires convolution with mask
  // (p*q elements).
  return 2*n*max_moment;
}

//----------------------------------------------------------------------
void moments_protocol(const uint n, const uint max_moment,
		      const double lower, const double upper) {
  const uint trials = 5;
  timer_418 t0, t1;
  double *run_times;

  // Place to store the data.
  float *x;
  double *mom_gpu;
  double *mom_cpu;

  // Allocate the vectors.
  NewArrayCPU(float, n, x);
  NewArrayCPU(double, max_moment, mom_gpu);
  NewArrayCPU(double, max_moment, mom_cpu);
  NewArrayCPU(double, trials, run_times);

  // Fill the initial vector.
  rand_float_vector(x, n, lower, upper);


  ///*
  // Pointers to the data on the GPU
  //float *d_x;
  //double *d_moments;
  // Allocate memory on the GPU and copy over input data.
  //data_to_gpu(x, n, max_moment, &d_x, &d_moments);
  // Compute the moments.
  //compute_all_gpu(d_x, n, max_moment, d_moments);
  // Make sure the GPU calculations are complete.
  //after_run_gpu();
  // Copy back the results from the GPU and deallocate the memory on the GPU.
  //data_from_gpu(mom_gpu, max_moment, d_x, d_moments);

  float *d_x;
  double *d_moments;
  

  printf("\n\nGPU:");

  for(uint i = 0; i < trials; i++) {
    timestamp(&t0);
    // Allocate memory on the GPU and copy over input data.
    data_to_gpu(x, n, max_moment, &d_x, &d_moments);
    // Compute the moments.
    compute_all_gpu(d_x, n, max_moment, d_moments);
    // Make sure the GPU calculations are complete.
    after_run_gpu();
    // Copy back the results from the GPU and deallocate the memory on the GPU.
    data_from_gpu(mom_gpu, max_moment, d_x, d_moments);

    timestamp(&t1);
    run_times[i] = elapsed_seconds(t1, t0);
    printf(" %5.2e", run_times[i]);
  }
  double median_time_gpu = median_time(run_times, trials);
  printf("\n  -- median %5.2e, flops %5.2e",
     median_time_gpu, total_fp_ops(n, max_moment) / median_time_gpu);

  // Show the results.
  for(uint j = 0; j < max_moment; j++) {
    printf("\nMoment %d: %f", j+1, mom_gpu[j]);
  }
  //*/

  // Compute the moments on the CPU.
  compute_all_cpu(x, n, max_moment, mom_cpu);

  printf("\n\nCPU:");

  for(uint i = 0; i < trials; i++) {
    timestamp(&t0);
    compute_all_cpu(x, n, max_moment, mom_cpu);
    timestamp(&t1);
    run_times[i] = elapsed_seconds(t1, t0);
    printf(" %5.2e", run_times[i]);
  }
  double median_time_cpu = median_time(run_times, trials);
  printf("\n  -- median %5.2e, flops %5.2e",
     median_time_cpu, total_fp_ops(n, max_moment) / median_time_cpu);

  // Show the results in CPU
  for(uint j = 0; j < max_moment; j++) {
    printf("\nMoment %d: %f", j+1, mom_cpu[j]);
  }

  // Free up all the CPU memory resources.
  free(x);
  free(mom_gpu);
  free(mom_cpu);
}


//----------------------------------------------------------------------
int main(int argc, char **argv) {

  uint n_default = 40000000, max_moment_default = 7;
  double lower_default = 0.0, upper_default = 1.0; 

  printf("Computing %d moments for %d samples in range [ %f, %f ]",
	 max_moment_default, n_default, lower_default, upper_default);
  moments_protocol(n_default, max_moment_default, 
		   lower_default, upper_default);
  printf("\n\n");
}
