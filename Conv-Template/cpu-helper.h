#include <stdlib.h>
#include <assert.h>

#ifndef CPU_HELPER
#define CPU_HELPER

//--------------------------------------------------------------------------
// Convenient allocation macros.
#define NewArrayCPU(Type, N, Host_P) assert(Host_P = ((Type *)(malloc(N * sizeof(Type)))))

//--------------------------------------------------------------------------
// Fortran column major indexing.  We don't actually need the number
// of columns (n), but we'll include it as an argument anyway for when
// we do C-style (row major) indexing.
#define IDX2F(i,j,m,n) (((j)*(m))+(i))

// C-style row major indexing.  We don't actually need the number
// of rows (m), but we'll include it as an argument anyway for when
// we do Fortran-style (column major) indexing.
#define IDX2C(i,j,m,n) (((i)*(n))+(j))

#endif

