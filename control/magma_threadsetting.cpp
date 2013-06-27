/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Azzam Haidar
*/
#include "common_magma.h"

/***************************************************************************//**
 * switch lapack thread_num initialization
 **/
#if defined(MAGMA_WITH_MKL)
#include <mkl_service.h>
#include <omp.h>
#endif

#if defined(MAGMA_WITH_ACML)
#include <omp.h>
#endif

#if defined(_OPENMP) 
#include <omp.h>
#endif

/////////////////////////////////////////////////////////////
extern "C"
void magma_setlapack_numthreads(magma_int_t num_threads)
{
#if defined(MAGMA_WITH_MKL)
    mkl_set_num_threads( num_threads );
#endif
#if defined(MAGMA_WITH_ACML)
    omp_set_num_threads( num_threads );
#endif
#if defined(_OPENMP) || defined(MAGMA_WITH_MKL)
    omp_set_num_threads( num_threads );
#endif
}
/////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////
extern "C"
magma_int_t magma_get_numthreads()
{
    /* determine the number of threads */
    magma_int_t threads = 0;
    char *env;
    // First check MKL_NUM_THREADS if MKL is used
#if defined(MAGMA_WITH_MKL)
    env = getenv("MKL_NUM_THREADS");
    if (env != NULL)
        threads = atoi(env);
#endif
    // Second check OMP_NUM_THREADS
    if (threads < 1){
        env = getenv("OMP_NUM_THREADS");
        if (env != NULL)
            threads = atoi(env);
    }
    // Third use the number of CPUs
    if (threads < 1)
        threads = sysconf(_SC_NPROCESSORS_ONLN);
    // Fourth use one thread
    if (threads < 1)
        threads = 1;

    return threads;
}
/////////////////////////////////////////////////////////////
