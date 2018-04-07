#include "cublas_v2.h"

#ifndef CUDA_HELPER
#define CUDA_HELPER

//--------------------------------------------------------------------------
// Convenient allocation macros.
#define NewArrayGPU(Type, N, Dev_P) cudaTry(cudaMalloc((void**)(&Dev_P), (N * sizeof(Type))))

//--------------------------------------------------------------------------
// function and macro to help diagnose cuBLAS failures.

#define cublasTry(cublasStatus) _cublasTry(cublasStatus, __FILE__, __LINE__)

const char *cublasGetErrorString(cublasStatus_t status);
void _cublasTry(cublasStatus_t cublasStatus, 
		const char *fileName, int lineNumber);


//--------------------------------------------------------------------------
// function and macro to help diagnose CUDA failures.
#define cudaTry(cudaStatus) _cudaTry(cudaStatus, __FILE__, __LINE__)

void _cudaTry(cudaError_t cudaStatus, const char *fileName, int lineNumber);

#endif
