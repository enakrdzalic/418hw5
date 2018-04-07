#include <stdio.h>
#include "cuda-helper.h"

//--------------------------------------------------------------------------
// functions to help diagnose cuBLAS failures.
const char *cublasGetErrorString(cublasStatus_t status) {
  switch(status) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  default:
    return "UNKNOWN!!!";
  }
}

void _cublasTry(cublasStatus_t cublasStatus, 
	       const char *fileName, int lineNumber) {
  if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "%s in %s line %d\n",
	    cublasGetErrorString(cublasStatus), fileName, lineNumber);
    exit(EXIT_FAILURE);
  }
}

//--------------------------------------------------------------------------
// function to help diagnose CUDA failures.

void _cudaTry(cudaError_t cudaStatus, const char *fileName, int lineNumber) {
  if(cudaStatus != cudaSuccess) {
    fprintf(stderr, "%s in %s line %d\n",
        cudaGetErrorString(cudaStatus), fileName, lineNumber);
    exit(1);
  }
}
