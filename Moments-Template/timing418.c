// Create some generic timing functions so that we can easily swap
// between choices.

#include <stdlib.h>
#include "cpu-helper.h"
#include "timing418.h"

//----------------------------------------------------------------------
#ifdef GetTimeOfDay

void timestamp(timer_418 *t) {
  gettimeofday(t, NULL);
}

double elapsed_seconds(timer_418 t1, timer_418 t0) {
  return (t1.tv_sec - t0.tv_sec) + 1e-6 * (t1.tv_usec - t0.tv_usec);
}

#endif

//----------------------------------------------------------------------
#ifdef RUsage

void timestamp(timer_418 *t) {
  getrusage(RUSAGE_SELF, t);
}

double elapsed_seconds(timer_418 t1, timer_418 t0) {
  return ((t1.ru_utime.tv_sec - t0.ru_utime.tv_sec)
	  + 1e-6 * (t1.ru_utime.tv_usec - t0.ru_utime.tv_usec));
}

#endif

//----------------------------------------------------------------------
// Used to sort the timing runs.
int compare_doubles(const void *a, const void *b) {
  double *x = (double *) a;
  double *y = (double *) b;

  if (*x < *y) 
    return -1;
  else if (*x > *y) 
    return 1; 
  else
    return 0;
}

//----------------------------------------------------------------------
double mean_time(double *run_times, int num_runs) {
  double sum_times = 0.0;
  for(int i = 0; i < num_runs; i++)
    sum_times = run_times[i];
  return sum_times / num_runs;
}

//----------------------------------------------------------------------
double median_time(double *run_times, int num_runs) {

  // We'll sort the times, which means we need a local copy.
  double *rt_local;
  NewArrayCPU(double, num_runs, rt_local);
  for(int i = 0; i < num_runs; i++)
    rt_local[i] = run_times[i];

  qsort(rt_local, num_runs, sizeof(double), compare_doubles);

  // Rule for the median depends on whether the number of runs is even or odd.
  double median = 0.0;
  if(num_runs % 2 == 1)
    // Odd number of runs.
    median = rt_local[num_runs / 2];
  else
    // Even number of runs.
    median = 0.5 * (rt_local[num_runs / 2 - 1] + rt_local[num_runs / 2]);

  free(rt_local);
  return median;
}
