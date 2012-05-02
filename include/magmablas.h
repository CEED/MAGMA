/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011
*/

#ifndef _MAGMABLAS_
#define _MAGMABLAS_

typedef int magma_int_t;
typedef int magma_err_t;
typedef void* magma_devptr;

#include <cublas.h>
#include <cuda.h>

#include "magmablas_z.h"
#include "magmablas_c.h"
#include "magmablas_d.h"
#include "magmablas_s.h"
#include "magmablas_zc.h"
#include "magmablas_ds.h"

#if (GPUSHMEM < 200)  
  #define magmablas_zgemm cublasZgemm
#endif
#define magmablas_cgemm cublasCgemm

#endif
