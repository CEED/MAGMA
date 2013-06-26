/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Azzam Haidar
*/
#include "common.h"
/***************************************************************************//**
 * switch lapack thread_num initialization
 **/
#if defined(MAGMA_WITH_MKL)
#include <mkl_service.h>
#endif
#if defined(MAGMA_WITH_ACML)
#include <omp.h>
#endif

void magma_setlapack_numthreads(int num_threads)
{
#if defined(MAGMA_WITH_MKL)
    mkl_set_num_threads( num_threads );
#endif
#if defined(MAGMA_WITH_ACML)
    omp_set_num_threads( num_threads );
#endif
}










