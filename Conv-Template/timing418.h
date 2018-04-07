// Header file for some generic timing functions.

#ifndef TIMING418
#define TIMING418

//----------------------------------------------------------------------
#ifdef GetTimeOfDay

#include <sys/time.h>

typedef struct timeval timer_418;

void timestamp(timer_418 *t);
double elapsed_seconds(timer_418 t1, timer_418 t0);

#endif

//----------------------------------------------------------------------
#ifdef RUsage

#include <sys/time.h>
#include <sys/resource.h>

typedef struct rusage timer_418;

void timestamp(timer_418 *t);
double elapsed_seconds(timer_418 t1, timer_418 t0);

#endif

//----------------------------------------------------------------------
// Some which are the same no matter which type of timing is used.
double mean_time(double *run_times, int num_runs);
double median_time(double *run_times, int num_runs);

#endif
